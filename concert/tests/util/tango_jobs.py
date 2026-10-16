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

import tango
from tango.server import Device, attribute, command

from concert.coroutines.base import run_in_executor


LOG = logging.getLogger(__name__)
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
# RPC replies and event snapshots arrive independently. Use this order to keep
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


class JobDevice(Device):
    """Small in-memory registry with one admitted background job at a time."""

    green_mode = tango.GreenMode.Asyncio
    history_limit = 16

    async def init_device(self):
        # Init can be called again on a live device. Finish cancellation before
        # discarding its resources or publishing a new server-instance ID.
        await self._stop_jobs()
        await super().init_device()
        # Every Init creates a new ID namespace. A retry from before Init must
        # not silently execute an old request against the newly initialized device.
        self._instance = str(uuid.uuid4())
        self._records = {}
        # Active tasks keep execution alive; records and requests outlive them
        # to support result retrieval and duplicate-start detection.
        self._tasks = {}
        self._requests = {}
        self._revision = 0
        self._emit_events = True
        # Include inherited commands, but only those explicitly opted in by
        # @job_command. Arbitrary device methods cannot be dispatched as jobs.
        self._operations = {
            name: getattr(type(self), name)._job_spec
            for name in dir(type(self))
            if hasattr(getattr(type(self), name), "_job_spec")
        }
        # Declare manually pushed change events and disable Tango's automatic
        # change detection: _publish decides when a status update is emitted.
        self.set_change_event("jobs", True, False)

    async def _stop_jobs(self):
        running = getattr(self, "_tasks", {})
        tasks = list(running.values())
        for job_id, task in running.items():
            # A second cancel() could interrupt a coroutine's asynchronous cleanup.
            if not task.done() and self._records[job_id]["state"] != "CANCELLING":
                self._records[job_id]["state"] = "CANCELLING"
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def delete_device(self):
        await self._stop_jobs()

    @attribute(dtype=str)
    def jobs(self):
        return self._snapshot()

    def _snapshot(self):
        # Results and tracebacks are only transferred by GetJob.
        return json.dumps({
            "version": 1,
            "instance": self._instance,
            "revision": self._revision,
            "jobs": [{k: v for k, v in record.items() if k not in ("result", "error")}
                     for record in self._records.values()],
        })

    def _publish(self):
        # Count state publications even when test fault injection suppresses
        # delivery. Revisions are comparable only within one server instance.
        self._revision += 1
        finished = [key for key, record in self._records.items()
                    if record["state"] in TERMINAL]
        # Bound completed history without ever evicting active work. Removing a
        # request also ends the duplicate-start guarantee for that ID.
        for key in finished[:-self.history_limit]:
            del self._records[key]
            del self._requests[key]
        if self._emit_events:
            try:
                self.push_change_event("jobs", self._snapshot())
            except tango.DevFailed:
                # Completion remains queryable even when publishing fails.
                LOG.exception("Could not publish job status")

    @command(dtype_in=str, dtype_out=str)
    async def StartJob(self, payload):
        request = json.loads(payload)
        if request.get("version") != 1:
            raise ValueError("Unsupported protocol version")
        job_id = request["id"]
        instance, token = job_id.split("/")
        uuid.UUID(token)
        if instance != self._instance:
            raise ValueError("Server instance changed")
        # Check retries before admission: the original job may still hold the
        # busy slot. Reusing its ID is safe only for the exact same request.
        if job_id in self._requests:
            if self._requests[job_id] != request:
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
            # Numeric inputs in this test device are durations, not general Tango
            # numbers. Bound them and reject bool (which is an int subclass).
            valid = isinstance(arg, str) if dtype is str else (
                type(arg) in (int, float) and 0 <= arg <= 60
            )
            if not valid:
                raise TypeError("Invalid command argument")
            args = (dtype(arg),)
        # Calling the coroutine directly bypasses Tango's normal command
        # dispatch, so apply its conventional admission hook explicitly.
        allowed = getattr(self, f"is_{name}_allowed", None)
        if allowed is not None:
            permit = allowed()
            if hasattr(permit, "__await__"):
                permit = await permit
            if not permit:
                raise RuntimeError("Command is not allowed")
        # Reserve admission without yielding, before acknowledging the start.
        if self._tasks:
            raise RuntimeError("Device busy")
        record = {"id": job_id, "command": name, "state": "PENDING",
                  "submitted": time.time()}
        self._records[job_id] = record
        self._requests[job_id] = request
        task = asyncio.create_task(self._execute(record, func, args))
        self._tasks[job_id] = task
        task.add_done_callback(functools.partial(self._finished, job_id))
        self._publish()
        return job_id

    async def _execute(self, record, func, args):
        record.update(state="RUNNING", started=time.time())
        self._publish()
        result = await func(self, *args)
        # Serialization errors must be job failures, not broken result reads.
        json.dumps(result, allow_nan=False)
        return result

    def _finished(self, job_id, task):
        # A done callback also covers cancellation before _execute even starts.
        # task.result() retrieves exceptions and records the actual outcome:
        # requesting cancellation does not guarantee the coroutine obeyed it.
        record = self._records[job_id]
        try:
            record.update(result=task.result(), state="SUCCEEDED")
        except asyncio.CancelledError:
            record["state"] = "CANCELLED"
        except Exception as exc:
            record.update(state="FAILED", error={
                "type": type(exc).__name__, "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            })
        record["finished"] = time.time()
        del self._tasks[job_id]
        self._publish()

    @command(dtype_in=str, dtype_out=str)
    def GetJob(self, job_id):
        # Missing history means the outcome is unknown, not that execution failed.
        # This includes expired IDs and IDs belonging to an earlier server instance.
        return json.dumps(self._records.get(job_id, {
            "id": job_id, "state": "UNAVAILABLE", "instance": self._instance,
        }))

    @command(dtype_in=str, dtype_out=str)
    def CancelJob(self, job_id):
        record = self._records[job_id]
        # Acknowledge the request immediately. Only _finished may report CANCELLED,
        # once the task has exited; repeated requests must not interrupt cleanup.
        if record["state"] not in TERMINAL and record["state"] != "CANCELLING":
            record["state"] = "CANCELLING"
            self._tasks[job_id].cancel()
            self._publish()
        return record["state"]


class JobTestDevice(JobDevice):
    """Controllable operations for integration tests; no hardware required."""

    async def init_device(self):
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

    @job_command(dtype_in=str, dtype_out=str)
    async def immediate(self, value):
        self._calls += 1
        return value

    @job_command(dtype_in=float, dtype_out=float)
    async def slow(self, seconds):
        self._calls += 1
        await asyncio.sleep(seconds)
        return seconds

    @job_command()
    async def fail(self):
        raise ValueError("deliberate test failure")

    @job_command(dtype_in=float)
    async def cleanup(self, seconds):
        self._cleaned_up = False
        try:
            await asyncio.sleep(seconds)
        finally:
            await asyncio.sleep(0.2)
            self._cleaned_up = True

    @job_command(dtype_in=float)
    async def blocking(self, seconds):
        # Concert's helper waits for the worker thread on cancellation. This
        # deliberately tests that the job stays CANCELLING until that work ends.
        await run_in_executor(time.sleep, seconds)

    @job_command(dtype_in=float, dtype_out=str)
    async def suppress_cancel(self, seconds):
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return "completed despite cancellation"
        return "completed"

    @job_command(dtype_out=str)
    async def bad_result(self):
        return object()

    @job_command()
    async def forbidden(self):
        raise AssertionError("Admission check was bypassed")

    def is_forbidden_allowed(self):
        return False


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
    def __init__(self, owner, job_id):
        self.id = job_id
        self._state = "PENDING"
        self._owner = owner
        self._outcome = asyncio.get_running_loop().create_future()

    @property
    def state(self):
        """Latest known status, including events received while asyncio is idle.

        This is a local cache read, not a network request. Lost events are
        reconciled by the watcher when the client's event loop runs.
        """
        if self._state not in TERMINAL and self._state != "UNAVAILABLE":
            observed = self._owner._event_state(self.id)
            if observed is not None and STATE_ORDER[observed] >= STATE_ORDER[self._state]:
                self._state = observed
        return self._state

    async def _wait(self):
        # Store errors as values so abandoned handles don't cause unhandled
        # Future exception warnings. Each waiter raises the recorded outcome.
        # Each await gets a cancellable wait on the same outcome. Timing out one
        # waiter must not cancel the outcome needed by other or future waiters.
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
        return await self._owner.proxy.CancelJob(self.id)


class JobDeviceProxy:
    """Test-only adapter; start/attach initialize monitoring lazily.

    Call close() when finished. Ordinary proxy calls do not initialize monitoring.
    """

    def __init__(self, proxy, *, reconcile_interval=30, recovery_timeout=10):
        self.proxy = proxy
        self.reconcile_interval = reconcile_interval
        self.recovery_timeout = recovery_timeout
        self._handles = {}
        self._wake = asyncio.Event()
        self._subscription = None
        self._monitor = None
        self._initialization = None
        self._close_lock = asyncio.Lock()
        # This is the only state shared with Tango's callback thread. The other
        # fields belong to the client's asyncio loop; an asyncio lock would not
        # protect this cache against a callback running on another thread.
        self._snapshot_lock = threading.Lock()
        self._event_snapshot = None
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.proxy, name)

    async def __aenter__(self):
        await self._ensure_started()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def _ensure_started(self):
        if self._closed:
            raise RuntimeError("Client is closed")
        if self._initialization is None:
            # No yield before storing the task: concurrent first calls share it.
            self._initialization = asyncio.create_task(self._initialize())
            self._initialization.add_done_callback(self._initialization_done)
        # Cancelling a caller must not cancel shared setup or lose an event
        # subscription whose registration is still executing inside PyTango.
        await asyncio.shield(self._initialization)
        if self._closed:
            raise RuntimeError("Client is closed")

    def _initialization_done(self, task):
        # Retrieve failures even if all callers cancelled their local waits.
        if task.cancelled() or task.exception() is not None:
            # Allow a later start/attach to retry setup after a transient failure.
            self._initialization = None

    async def _initialize(self):
        self._loop = asyncio.get_running_loop()
        # Retry cleanup first if an earlier unsubscribe failed.
        if self._subscription is not None:
            await self.proxy.unsubscribe_event(self._subscription)
            self._subscription = None
        # Stateless subscriptions allow Tango to retry connecting. The spelling
        # changed in newer PyTango; keep the installed 9.5 API working as well.
        options = ({"sub_mode": tango.EventSubMode.Stateless}
                   if hasattr(tango, "EventSubMode") else {"stateless": True})
        # Synchronous callbacks run on Tango's event thread, even while terminal
        # IPython has paused its loop. Offload the subscription RPC itself.

        def subscribe():
            return self.proxy.subscribe_event(
                "jobs", tango.EventType.CHANGE_EVENT, self._on_event,
                green_mode=tango.GreenMode.Synchronous, **options,
            )

        self._subscription = await run_in_executor(subscribe)
        try:
            snapshot = json.loads((await self.proxy.read_attribute("jobs")).value)
            self._instance = snapshot["instance"]
            self._monitor = asyncio.create_task(self._watch())
        except BaseException:
            await self.proxy.unsubscribe_event(self._subscription)
            self._subscription = None
            raise

    def _on_event(self, event):
        # Only the snapshot cache is touched from Tango's thread. Futures,
        # handles and result queries remain owned by the asyncio loop.
        if self._closed:
            return
        if not event.err and event.attr_value is not None:
            try:
                snapshot = json.loads(event.attr_value.value)
                with self._snapshot_lock:
                    previous = self._event_snapshot
                    # Ignore duplicate/older snapshots from the same instance.
                    # A different instance has its own revision counter.
                    if (previous is None or snapshot["instance"] != previous["instance"]
                            or snapshot["revision"] > previous["revision"]):
                        self._event_snapshot = snapshot
            except (TypeError, ValueError, KeyError):
                LOG.exception("Invalid job status event")
        # Error events also trigger reconciliation. Never call Event.set()
        # directly from the Tango thread.
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            # A callback already in flight can race client or event-loop shutdown.
            if not self._closed and not self._loop.is_closed():
                raise

    def _event_state(self, job_id):
        with self._snapshot_lock:
            snapshot = self._event_snapshot
            if snapshot is None:
                return None
            if job_id.split("/", 1)[0] != snapshot["instance"]:
                return "UNAVAILABLE"
            for record in snapshot["jobs"]:
                if record["id"] == job_id:
                    return record["state"]
        # Absence alone is inconclusive: the snapshot may predate this job.
        # GetJob, rather than this cache lookup, determines whether it is missing.
        return None

    async def start(self, name, *args, job_id=None):
        if len(args) > 1:
            raise TypeError("Tango commands take at most one input argument")
        await self._ensure_started()
        # Generate the ID before the RPC, so even a lost acknowledgement leaves
        # the caller an ID to attach to or reuse when retrying this exact request.
        job_id = job_id or f"{self._instance}/{uuid.uuid4()}"
        request = {"version": 1, "id": job_id, "command": name}
        if args:
            request["arg"] = args[0]
        try:
            await self.proxy.StartJob(json.dumps(request))
        except tango.DevFailed as exc:
            # Preserve the ID even when acceptance is ambiguous. Callers can
            # inspect the cause, attach, or retry this exact request and ID.
            if any(error.reason in CONNECTION_ERRORS for error in exc.args):
                raise JobConnectionError(job_id, "StartJob did not acknowledge the job") from exc
            raise
        return await self.attach(job_id)

    async def attach(self, job_id):
        await self._ensure_started()
        job = self._handles.get(job_id)
        if job is None or job._outcome.done():
            job = self._handles[job_id] = RemoteJob(self, job_id)
        # The subscription already exists; this read closes the completion race.
        try:
            await self._refresh(job)
        except tango.DevFailed:
            # Keep the handle: a failed status read does not imply a failed job.
            # The observer retries and applies the connection recovery deadline.
            self._wake.set()
        return job

    async def _refresh(self, job):
        if job._outcome.done():
            return
        record = json.loads(await self.proxy.GetJob(job.id))
        # Attachment and the event observer can have reads in flight together.
        # Never settle twice or overwrite a terminal state with an older read.
        if job._outcome.done():
            return
        state = record["state"]
        job._state = state
        # An event may already report a later state than this RPC response.
        # Only settle with the result/error belonging to the response itself.
        if state == "UNAVAILABLE":
            job._outcome.set_result(JobUnavailable(job.id))
        elif state in TERMINAL:
            job._outcome.set_result(record)
        if job._outcome.done():
            # The caller's handle keeps its outcome; the observer no longer needs
            # to retain it or query this job on every subsequent event.
            self._handles.pop(job.id, None)

    async def _watch(self):
        disconnected_at = None
        while True:
            interval = (self.reconcile_interval if disconnected_at is None
                        else min(self.reconcile_interval, 0.1))
            try:
                # Events are the fast path; the timeout recovers a silently lost
                # final event. An Event coalesces bursts into one reconciliation.
                await asyncio.wait_for(self._wake.wait(), interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                # _refresh removes finished handles, and attach can add new ones
                # while an RPC is awaited, so iterate over a stable copy.
                for job in list(self._handles.values()):
                    await self._refresh(job)
                disconnected_at = None
            except tango.DevFailed as exc:
                now = asyncio.get_running_loop().time()
                if disconnected_at is None:
                    disconnected_at = now
                if now - disconnected_at >= self.recovery_timeout:
                    # This deadline limits loss of contact, not execution time.
                    # Fail only the local waits; the server job may still be running.
                    for job in self._handles.values():
                        if not job._outcome.done():
                            job._outcome.set_result(JobConnectionError(job.id, str(exc)))
                    self._handles.clear()
                # Bound retries even if an event source keeps reporting errors.
                await asyncio.sleep(min(self.reconcile_interval, 0.1))

    async def close(self):
        """Release monitoring, without cancelling server jobs; safe to repeat."""
        self._closed = True
        async with self._close_lock:
            # Setup may still be registering a subscription. Let it finish so
            # close knows its ID and can release it; serialize concurrent closes.
            if self._initialization is not None:
                try:
                    await asyncio.shield(self._initialization)
                except Exception:
                    pass  # Setup cleans up its subscription on failure.
            if self._monitor is not None:
                self._monitor.cancel()
                await asyncio.gather(self._monitor, return_exceptions=True)
            try:
                if self._subscription is not None:
                    await self.proxy.unsubscribe_event(self._subscription)
                    self._subscription = None
            finally:
                for job in self._handles.values():
                    if not job._outcome.done():
                        job._outcome.set_result(JobConnectionError(job.id, "Client closed"))
                self._handles.clear()
