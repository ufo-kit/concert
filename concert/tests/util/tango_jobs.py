"""Experimental job protocol and device; deliberately not a production API.

See TANGO_JOBS.md in this directory for scope and how to run the tests.
"""
import asyncio
import functools
import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass

import tango
from tango.server import Device, attribute, command

from concert.coroutines.base import run_in_executor


LOG = logging.getLogger(__name__)
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
# RPC replies and status events arrive independently. Use this order to keep
# a cached status from moving backwards; terminal outcomes are latched separately.
STATE_ORDER = {"PENDING": 0, "RUNNING": 1, "CANCELLING": 2,
               "SUCCEEDED": 3, "FAILED": 3, "CANCELLED": 3, "UNAVAILABLE": 4}
CONNECTION_ERRORS = {"API_CommunicationFailed", "API_CantConnectToDevice",
                     "API_DeviceTimedOut", "API_CorbaException"}


def job_command(*, dtype_in=None, dtype_out=None):
    """Keep the ordinary command and register its coroutine for StartJob.

    This experiment only accepts void, string and float command inputs.
    """
    if dtype_in not in (None, str, float):
        raise TypeError("Unsupported prototype input type")

    def decorate(func):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("Job commands must be coroutine functions")
        wrapped = command(dtype_in=dtype_in, dtype_out=dtype_out)(func)
        # Keep the original coroutine alongside PyTango's command wrapper:
        # StartJob schedules it locally rather than making another Tango call.
        wrapped._job_spec = (func, dtype_in)
        return wrapped

    return decorate


@dataclass
class Job:
    """One retained request and outcome; release the task after completion."""

    request: dict
    state: str = "PENDING"
    task: asyncio.Task | None = None
    result: object = None
    error: dict | None = None

    def record(self):
        return {"id": self.request["id"], "command": self.request["command"],
                "state": self.state, "result": self.result, "error": self.error}

    def cancel(self):
        # Repeated cancellation must not interrupt asynchronous cleanup.
        if self.task is not None and not self.task.done() and self.state != "CANCELLING":
            self.state = "CANCELLING"
            self.task.cancel()


class JobDevice(Device):
    """Small in-memory registry with one admitted background job at a time."""

    green_mode = tango.GreenMode.Asyncio
    history_limit = 16

    async def init_device(self):
        # Init can be called again on a live device. Finish cancellation before
        # discarding its resources and job history.
        await self._stop_jobs()
        await super().init_device()
        self._jobs = {}
        self._last_change = json.dumps({"id": None, "state": None})
        # Include inherited commands, but only those explicitly opted in by
        # @job_command. Arbitrary device methods cannot be dispatched as jobs.
        self._operations = {
            name: getattr(type(self), name)._job_spec
            for name in dir(type(self))
            if hasattr(getattr(type(self), name), "_job_spec")
        }
        # Declare manually pushed change events and disable Tango's automatic
        # change detection: _publish decides when a status update is emitted.
        self.set_change_event("job_changed", True, False)

    async def _stop_jobs(self):
        tasks = []
        for job in getattr(self, "_jobs", {}).values():
            if job.task is not None:
                tasks.append(job.task)
                job.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def delete_device(self):
        await self._stop_jobs()

    @attribute(dtype=str)
    def job_changed(self):
        return self._last_change

    def _publish(self, job):
        # Publish only the changed job. GetJob remains the source of results and
        # missed updates; reading this attribute does not replay earlier events.
        self._last_change = json.dumps({
            "id": job.request["id"], "state": job.state,
        })
        try:
            self.push_change_event("job_changed", self._last_change)
        except tango.DevFailed:
            LOG.exception("Could not publish job status")

    @command(dtype_in=str, dtype_out=str)
    async def StartJob(self, payload):
        request = json.loads(payload)
        if request.get("version") != 1:
            raise ValueError("Unsupported protocol version")
        job_id = request["id"]
        uuid.UUID(job_id)
        # Check retries before admission: the original job may still hold the
        # busy slot. Reusing its ID is safe only for the exact same request.
        if job_id in self._jobs:
            if self._jobs[job_id].request != request:
                raise ValueError("Job ID reused with different arguments")
            return job_id
        name = request["command"]
        if name not in self._operations:
            raise ValueError("Command is not registered for jobs")
        func, dtype = self._operations[name]
        if dtype is None:
            if "arg" in request:
                raise TypeError("Command takes no argument")
            args = ()
        else:
            arg = request["arg"]
            valid = isinstance(arg, str) if dtype is str else type(arg) in (int, float)
            if not valid:
                raise TypeError("Invalid command argument")
            args = (dtype(arg),)
        # Reserve admission without yielding, before acknowledging the start.
        if any(job.task is not None for job in self._jobs.values()):
            raise RuntimeError("Device busy")
        job = self._jobs[job_id] = Job(request)
        job.task = asyncio.create_task(self._execute(job, func, args))
        job.task.add_done_callback(functools.partial(self._finished, job))
        self._publish(job)
        return job_id

    async def _execute(self, job, func, args):
        job.state = "RUNNING"
        self._publish(job)
        result = await func(self, *args)
        # Serialization errors must be job failures, not broken result reads.
        json.dumps(result, allow_nan=False)
        return result

    def _finished(self, job, task):
        # A done callback also covers cancellation before _execute even starts.
        # task.result() retrieves exceptions and records the actual outcome:
        # requesting cancellation does not guarantee the coroutine obeyed it.
        try:
            job.result = task.result()
            job.state = "SUCCEEDED"
        except asyncio.CancelledError:
            job.state = "CANCELLED"
        except Exception as exc:
            job.state = "FAILED"
            job.error = {
                "type": type(exc).__name__, "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
        job.task = None
        # Bound retained outcomes without evicting active work. Eviction also
        # ends the duplicate-start guarantee for that ID.
        finished = [key for key, item in self._jobs.items() if item.state in TERMINAL]
        for key in finished[:max(0, len(finished) - self.history_limit)]:
            del self._jobs[key]
        self._publish(job)

    @command(dtype_in=str, dtype_out=str)
    def GetJob(self, job_id):
        # Missing history means the outcome is unknown, not that execution failed.
        # This includes expired records and history lost on restart or Init.
        job = self._jobs.get(job_id)
        return json.dumps(job.record() if job else {
            "id": job_id, "state": "UNAVAILABLE",
        })

    @command(dtype_in=str, dtype_out=str)
    def CancelJob(self, job_id):
        job = self._jobs[job_id]
        # Acknowledge the request immediately. Only _finished may report CANCELLED,
        # once the task has exited; repeated requests must not interrupt cleanup.
        job.cancel()
        self._publish(job)
        return job.state


class JobTestDevice(JobDevice):
    """Controllable operations for integration tests; no hardware required."""

    async def init_device(self):
        self._emit_events = True
        await super().init_device()
        self._calls = 0
        self._cleaned_up = False

    @attribute(dtype=int)
    def calls(self):
        return self._calls

    @attribute(dtype=bool)
    def cleaned_up(self):
        return self._cleaned_up

    @command(dtype_out=str)
    def echo(self):
        return "responsive"

    @command(dtype_in=bool)
    def emit_events(self, enabled):
        self._emit_events = enabled

    def push_change_event(self, *args, **kwargs):
        # Fault injection belongs to the test device, not the job protocol.
        if self._emit_events:
            return super().push_change_event(*args, **kwargs)

    @staticmethod
    def _check_duration(seconds):
        if not 0 <= seconds <= 60:
            raise ValueError("Test duration must be between 0 and 60 seconds")

    @job_command(dtype_in=str, dtype_out=str)
    async def immediate(self, value):
        self._calls += 1
        return value

    @job_command(dtype_in=float, dtype_out=float)
    async def slow(self, seconds):
        self._check_duration(seconds)
        self._calls += 1
        await asyncio.sleep(seconds)
        return seconds

    @job_command()
    async def fail(self):
        raise ValueError("deliberate test failure")

    @job_command(dtype_in=float)
    async def cleanup(self, seconds):
        self._check_duration(seconds)
        self._cleaned_up = False
        try:
            await asyncio.sleep(seconds)
        finally:
            await asyncio.sleep(0.2)
            self._cleaned_up = True

    @job_command(dtype_in=float)
    async def blocking(self, seconds):
        self._check_duration(seconds)
        # Concert's helper waits for the worker thread on cancellation. This
        # deliberately tests that the job stays CANCELLING until that work ends.
        await run_in_executor(time.sleep, seconds)

    @job_command(dtype_in=float, dtype_out=str)
    async def suppress_cancel(self, seconds):
        self._check_duration(seconds)
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return "completed despite cancellation"
        return "completed"

    @job_command(dtype_out=str)
    async def bad_result(self):
        return object()


class RemoteJobError(RuntimeError):
    def __init__(self, job_id, error):
        self.job_id = job_id
        self.remote_error = error
        super().__init__(f"{job_id}: {error['type']}: {error['message']}")


class RemoteJobCancelled(RuntimeError):
    pass


class JobUnavailable(RuntimeError):
    pass


class JobConnectionError(ConnectionError):
    def __init__(self, job_id, message):
        self.job_id = job_id
        super().__init__(message)


class RemoteJob:
    """One awaitable remote operation, with its own status subscription.

    Use start(proxy, command, ...) or attach(proxy, job_id) to create a handle.
    Monitoring stops on completion; close() stops it early without cancelling
    the server operation. Ordinary Tango calls belong on the supplied proxy.
    """

    def __init__(self, proxy, job_id, *, reconcile_interval=30, recovery_timeout=10):
        self.id = job_id
        self.proxy = proxy
        self.reconcile_interval = reconcile_interval
        self.recovery_timeout = recovery_timeout
        self._loop = asyncio.get_running_loop()
        self._outcome = self._loop.create_future()
        self._state = "PENDING"
        # Tango's callback thread updates this even when IPython's loop is idle.
        self._state_lock = threading.Lock()
        self._wake = asyncio.Event()
        self._subscription = None
        self._monitor = None
        self._close_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def start(cls, proxy, name, *args, job_id=None,
                    reconcile_interval=30, recovery_timeout=10):
        """Submit one operation and return its handle, without waiting for it."""
        if len(args) > 1:
            raise TypeError("Tango commands take at most one input argument")
        if job_id is None:
            job_id = str(uuid.uuid4())
        request = {"version": 1, "id": job_id, "command": name}
        if args:
            request["arg"] = args[0]
        try:
            await proxy.StartJob(json.dumps(request))
            return await cls.attach(proxy, job_id, reconcile_interval=reconcile_interval,
                                    recovery_timeout=recovery_timeout)
        except tango.DevFailed as exc:
            # Preserve the ID when acceptance is ambiguous, so the caller can
            # attach or retry the exact request without duplicating execution.
            if any(error.reason in CONNECTION_ERRORS for error in exc.args):
                raise JobConnectionError(job_id, "Could not confirm or monitor the job") from exc
            raise

    @classmethod
    async def attach(cls, proxy, job_id, *, reconcile_interval=30, recovery_timeout=10):
        """Observe an existing job without starting another server operation."""
        job = cls(proxy, job_id, reconcile_interval=reconcile_interval,
                  recovery_timeout=recovery_timeout)
        try:
            await job._subscribe()
            # Subscribe before reading: even a job that finished before attachment
            # is recovered from history, without relying on its completion event.
            try:
                await job._refresh()
            except tango.DevFailed:
                job._wake.set()  # A failed read is not a failed server job.
            if job._outcome.done():
                await job.close()
            else:
                job._monitor = asyncio.create_task(job._watch())
        except BaseException:
            await job.close()
            raise
        return job

    async def _subscribe(self):
        options = ({"sub_mode": tango.EventSubMode.Stateless}
                   if hasattr(tango, "EventSubMode") else {"stateless": True})

        def subscribe():
            return self.proxy.subscribe_event(
                "job_changed", tango.EventType.CHANGE_EVENT, self._on_event,
                green_mode=tango.GreenMode.Synchronous, **options,
            )

        # Subscription is blocking, but callbacks must run on Tango's thread.
        # If attachment is cancelled, still collect the ID so close can release it.
        subscribing = asyncio.create_task(run_in_executor(subscribe))
        try:
            self._subscription = await asyncio.shield(subscribing)
        except asyncio.CancelledError:
            self._subscription = await subscribing
            raise

    @property
    def state(self):
        """Latest cached state; missed events are reconciled while asyncio runs."""
        with self._state_lock:
            return self._state

    def _observe_state(self, state):
        with self._state_lock:
            if (self._state not in TERMINAL
                    and STATE_ORDER[state] > STATE_ORDER[self._state]):
                self._state = state

    def _on_event(self, event):
        if self._closed:
            return
        if not event.err and event.attr_value is not None:
            try:
                change = json.loads(event.attr_value.value)
                if change["id"] == self.id:
                    self._observe_state(change["state"])
                else:
                    return  # Other jobs on the device are not this handle's concern.
            except (TypeError, ValueError, KeyError):
                LOG.exception("Invalid job status event")
        # Only cached state is touched above. Futures and RPCs belong to asyncio.
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            if not self._closed and not self._loop.is_closed():
                raise

    async def _refresh(self):
        if self._outcome.done():
            return
        record = json.loads(await self.proxy.GetJob(self.id))
        if self._outcome.done():
            return
        state = record["state"]
        self._observe_state(state)
        # An event may be ahead of this reply. Settle only with a terminal RPC
        # record, which includes the actual result or error, not the cached state.
        if state == "UNAVAILABLE" or state in TERMINAL:
            with self._state_lock:
                self._state = state
            self._outcome.set_result(JobUnavailable(self.id) if state == "UNAVAILABLE" else record)

    async def _watch(self):
        disconnected_at = None
        try:
            while not self._outcome.done():
                interval = (self.reconcile_interval if disconnected_at is None
                            else min(self.reconcile_interval, 0.1))
                try:
                    # Events are the fast path; timeout recovers silently lost events.
                    await asyncio.wait_for(self._wake.wait(), interval)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                try:
                    await self._refresh()
                    disconnected_at = None
                except tango.DevFailed as exc:
                    now = self._loop.time()
                    if disconnected_at is None:
                        disconnected_at = now
                    if now - disconnected_at >= self.recovery_timeout:
                        # This limits loss of contact, not the job's running time.
                        self._outcome.set_result(JobConnectionError(self.id, str(exc)))
                        break
                    await asyncio.sleep(min(self.reconcile_interval, 0.1))
        except Exception as exc:
            if not self._outcome.done():
                self._outcome.set_result(exc)
        finally:
            self._closed = True
            try:
                await self._unsubscribe()
            except tango.DevFailed:
                # Keep the subscription ID so an explicit close can retry cleanup.
                LOG.exception("Could not unsubscribe from job %s", self.id)

    async def _unsubscribe(self):
        async with self._close_lock:
            if self._subscription is not None:
                await self.proxy.unsubscribe_event(self._subscription)
                self._subscription = None

    async def close(self):
        """Stop local monitoring, without cancelling the remote operation."""
        if self._monitor is not None:
            # Do not interrupt cleanup if another close or completion got here first.
            if not self._closed and not self._monitor.done():
                self._monitor.cancel()
            self._closed = True
            await asyncio.gather(self._monitor, return_exceptions=True)
        self._closed = True
        try:
            await self._unsubscribe()
        finally:
            if not self._outcome.done():
                self._outcome.set_result(JobConnectionError(self.id, "Job monitoring closed"))

    async def _wait(self):
        # Errors are stored as values to avoid warnings for abandoned handles.
        # Cancelling one waiter must not cancel the outcome or another waiter.
        outcome = await asyncio.shield(self._outcome)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome["state"] == "FAILED":
            raise RemoteJobError(self.id, outcome["error"])
        if outcome["state"] == "CANCELLED":
            raise RemoteJobCancelled(self.id)
        return outcome["result"]

    def __await__(self):
        return self._wait().__await__()

    async def cancel_remote(self):
        """Request server cancellation; await this handle to wait for cleanup."""
        return await self.proxy.CancelJob(self.id)
