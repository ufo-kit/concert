"""Exercise the experimental protocol over real Tango RPC and change events."""
import asyncio
import json
import time
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

tango = pytest.importorskip("tango")
from tango.asyncio_executor import AsyncioExecutor  # noqa: E402
from tango.test_context import DeviceTestContext  # noqa: E402

from concert.tests.util.tango_jobs import (  # noqa: E402
    JobConnectionError, JobTestDevice, JobUnavailable,
    RemoteJob, RemoteJobCancelled, RemoteJobError,
)


class FaultyConnection:
    """Inject transport failures without changing server execution."""

    def __init__(self, proxy):
        self.proxy = proxy
        self.offline = False
        self.lose_start_reply = False

    def __getattr__(self, name):
        return getattr(self.proxy, name)

    async def GetJob(self, job_id):
        if self.offline:
            tango.Except.throw_exception("API_CommunicationFailed", "test outage", "test")
        return await self.proxy.GetJob(job_id)

    async def StartJob(self, request):
        result = await self.proxy.StartJob(request)
        if self.lose_start_reply:
            self.lose_start_reply = False
            tango.Except.throw_exception("API_DeviceTimedOut", "lost reply", "test")
        return result


class SetupConnection(FaultyConnection):
    """Count subscriptions and control when subscription setup finishes."""

    def __init__(self, proxy):
        super().__init__(proxy)
        self.subscriptions = 0
        self.unsubscriptions = 0
        self.subscribed = asyncio.Event()
        self.loop = asyncio.get_running_loop()
        self.release_subscription = threading.Event()
        self.release_subscription.set()

    def subscribe_event(self, *args, **kwargs):
        subscription = self.proxy.subscribe_event(*args, **kwargs)
        self.subscriptions += 1
        self.loop.call_soon_threadsafe(self.subscribed.set)
        self.release_subscription.wait()
        return subscription

    async def unsubscribe_event(self, subscription):
        await self.proxy.unsubscribe_event(subscription)
        self.unsubscriptions += 1


class TestTangoJobs(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # A separate process avoids mixing Tango's server and client executors.
        cls.context = DeviceTestContext(JobTestDevice, process=True, host="127.0.0.1")
        cls.context.start()

    @classmethod
    def tearDownClass(cls):
        cls.context.stop()

    async def asyncSetUp(self):
        self.proxy = self.make_proxy()
        await self.proxy.Init()

    def make_proxy(self):
        proxy = tango.DeviceProxy(
            self.context.get_device_access(), green_mode=tango.GreenMode.Asyncio,
            asyncio_executor=AsyncioExecutor(loop=asyncio.get_running_loop()),
        )
        proxy.set_timeout_millis(500)
        return proxy

    async def asyncTearDown(self):
        await self.proxy.Init()

    async def record(self, job):
        return json.loads(await self.proxy.GetJob(job.id))

    async def wait_state(self, job, state):
        async def wait():
            while (await self.record(job))["state"] != state:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait(), 2)

    async def start(self, name, *args, proxy=None, **options):
        options.setdefault("reconcile_interval", 0.05)
        options.setdefault("recovery_timeout", 0.15)
        job = await RemoteJob.start(self.proxy if proxy is None else proxy, name, *args, **options)
        self.addAsyncCleanup(job.close)
        return job

    async def attach(self, job_id, proxy=None, **options):
        options.setdefault("reconcile_interval", 0.05)
        options.setdefault("recovery_timeout", 0.15)
        job = await RemoteJob.attach(self.proxy if proxy is None else proxy, job_id, **options)
        self.addAsyncCleanup(job.close)
        return job

    async def test_single_job_releases_subscription_on_completion(self):
        connection = SetupConnection(self.proxy)
        job = await self.start("slow", 0.1, proxy=connection)
        self.assertFalse(hasattr(job, "echo"))
        self.assertEqual(await self.proxy.echo(), "responsive")
        self.assertEqual(await job, 0.1)
        await asyncio.wait_for(job._monitor, 2)
        self.assertIsNone(job._subscription)
        self.assertEqual(connection.subscriptions, 1)
        self.assertEqual(connection.unsubscriptions, 1)
        await asyncio.gather(job.close(), job.close())
        self.assertEqual(connection.unsubscriptions, 1)

    async def test_attachments_monitor_independently(self):
        connection = SetupConnection(self.proxy)
        first = await self.start("slow", 0.2, proxy=connection)
        second = await self.attach(first.id, proxy=connection)
        self.assertIsNot(first, second)
        self.assertEqual(connection.subscriptions, 2)
        await asyncio.gather(first.close(), first.close())
        with self.assertRaises(JobConnectionError):
            await first
        self.assertEqual(await second, 0.2)
        await second._monitor
        self.assertEqual(connection.unsubscriptions, 2)

    async def test_failed_start_does_not_create_subscription(self):
        connection = SetupConnection(self.proxy)
        with self.assertRaises(tango.DevFailed):
            await self.start("echo", proxy=connection)
        self.assertEqual(connection.subscriptions, 0)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 0)
        job = await self.start("immediate", "retry", proxy=connection)
        self.assertEqual(await job, "retry")
        await job.close()
        self.assertEqual(connection.subscriptions, connection.unsubscriptions)

    async def test_subscription_failure_after_start_preserves_job_id(self):
        class FailedSubscription(FaultyConnection):
            def subscribe_event(self, *args, **kwargs):
                tango.Except.throw_exception("API_CommunicationFailed", "offline", "test")

        with self.assertRaises(JobConnectionError) as caught:
            await self.start("slow", 0.2, proxy=FailedSubscription(self.proxy))
        job = await self.attach(caught.exception.job_id)
        self.assertEqual(await job, 0.2)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 1)

    async def test_close_does_not_interrupt_automatic_unsubscription(self):
        cleaning = asyncio.Event()
        release = asyncio.Event()

        class SlowUnsubscribe(SetupConnection):
            async def unsubscribe_event(self, subscription):
                cleaning.set()
                await release.wait()
                await super().unsubscribe_event(subscription)

        connection = SlowUnsubscribe(self.proxy)
        job = await self.start("slow", 0.1, proxy=connection)
        self.assertEqual(await job, 0.1)
        await asyncio.wait_for(cleaning.wait(), 2)
        closing = asyncio.create_task(job.close())
        try:
            await asyncio.sleep(0)
        finally:
            release.set()
        await asyncio.wait_for(closing, 2)
        self.assertEqual(connection.unsubscriptions, 1)

    async def test_cancelled_subscription_setup_releases_subscription(self):
        connection = SetupConnection(self.proxy)
        connection.release_subscription.clear()
        starting = asyncio.create_task(self.start("slow", 0.3, proxy=connection))
        try:
            await asyncio.wait_for(connection.subscribed.wait(), 2)
            starting.cancel()
            await asyncio.sleep(0)
        finally:
            connection.release_subscription.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 2)
        self.assertEqual(connection.subscriptions, 1)
        self.assertEqual(connection.unsubscriptions, 1)
        # Cancelling attachment must not cancel the accepted operation.
        change = json.loads((await self.proxy.read_attribute("job_changed")).value)
        attached = await self.attach(change["id"])
        self.assertEqual(await attached, 0.3)

    async def test_start_uses_plain_uuid_without_attribute_read(self):
        class NoAttributeRead(FaultyConnection):
            async def read_attribute(self, name):
                raise AssertionError("Starting a job must not read device attributes")

        job = await self.start("immediate", "result", proxy=NoAttributeRead(self.proxy))
        self.assertEqual(str(uuid.UUID(job.id)), job.id)
        self.assertEqual(await job, "result")

    async def test_short_rpc_timeout_and_responsive_device(self):
        job = await self.start("slow", 1.1)
        self.assertNotIn((await self.record(job))["state"], {"SUCCEEDED", "FAILED"})
        for _ in range(3):
            self.assertEqual(await self.proxy.echo(), "responsive")
            self.assertGreaterEqual((await self.proxy.read_attribute("calls")).value, 1)
        self.assertEqual(await asyncio.wait_for(job, 3), 1.1)
        self.assertEqual(self.proxy.get_timeout_millis(), 500)

    async def test_completion_uses_change_events(self):
        # A successful wait within 2 seconds cannot come from a 60-second poll.
        job = await self.start("slow", 0.2, reconcile_interval=60)
        self.assertEqual(await asyncio.wait_for(job, 2), 0.2)

    async def test_events_contain_only_one_job_and_ignore_late_updates(self):
        job = await self.start("slow", 0.3)
        running = json.loads((await self.proxy.read_attribute("job_changed")).value)
        self.assertEqual(set(running), {"id", "state"})
        self.assertEqual(running["id"], job.id)
        time.sleep(0.6)
        self.assertEqual(job.state, "SUCCEEDED")
        # An older notification must not undo the terminal event, even before
        # GetJob has fetched the result.
        event = SimpleNamespace(err=False, attr_value=SimpleNamespace(value=json.dumps(running)))
        job._on_event(event)
        self.assertEqual(job.state, "SUCCEEDED")
        unknown = dict(running, id=str(uuid.uuid4()))
        event.attr_value.value = json.dumps(unknown)
        job._on_event(event)
        self.assertEqual(job.state, "SUCCEEDED")
        self.assertEqual(await job, 0.3)
        self.assertEqual(job.state, "SUCCEEDED")

    async def test_state_updates_while_interactive_event_loop_is_idle(self):
        job = await self.start("slow", 0.2)
        # Terminal IPython stops its loop between async cells. Deliberately
        # block this loop to reproduce that, leaving Tango's threads running.
        time.sleep(0.6)
        self.assertEqual(job.state, "SUCCEEDED")
        self.assertEqual(await job, 0.2)

    async def test_cancelled_state_updates_while_loop_is_idle(self):
        job = await self.start("cleanup", 10)
        await self.wait_state(job, "RUNNING")
        await job.cancel_remote()
        time.sleep(0.6)
        self.assertEqual(job.state, "CANCELLED")
        with self.assertRaises(RemoteJobCancelled):
            await job

    async def test_terminal_event_does_not_settle_with_stale_rpc_result(self):
        job = await self.start("slow", 0.3)
        # Pause reconciliation without closing the event subscription.
        refreshed = asyncio.Event()
        job._refresh = AsyncMock(side_effect=refreshed.set)
        job._wake.set()
        await asyncio.wait_for(refreshed.wait(), 2)
        running_record = await self.proxy.GetJob(job.id)
        self.assertEqual(json.loads(running_record)["state"], "RUNNING")
        time.sleep(0.6)
        self.assertEqual(job.state, "SUCCEEDED")

        class StaleRead(FaultyConnection):
            async def GetJob(self, job_id):
                return running_record

        del job._refresh
        job.proxy = StaleRead(self.proxy)
        await job._refresh()
        self.assertEqual(job.state, "SUCCEEDED")
        self.assertFalse(job._outcome.done())
        job.proxy = self.proxy
        await job._refresh()
        self.assertEqual(await job, 0.3)

    async def test_immediate_completion_and_legacy_command(self):
        self.assertEqual(await self.proxy.immediate("legacy"), "legacy")
        for _ in range(5):
            job = await self.start("immediate", "result")
            self.assertEqual(await asyncio.wait_for(job, 2), "result")

    async def test_concurrent_completion_reads_settle_handle_once(self):
        finished = await self.start("immediate", "result")
        await finished
        response = await self.proxy.GetJob(finished.id)
        both_reading = asyncio.Event()

        class SimultaneousReads(FaultyConnection):
            reads = 0

            async def GetJob(self, job_id):
                self.reads += 1
                if self.reads == 2:
                    both_reading.set()
                await both_reading.wait()
                return response

        job = RemoteJob(SimultaneousReads(self.proxy), finished.id)
        self.addAsyncCleanup(job.close)
        await asyncio.wait_for(asyncio.gather(
            job._refresh(), job._refresh(),
        ), 2)
        self.assertEqual(await job, "result")

    async def test_remote_failure(self):
        job = await self.start("fail")
        with self.assertRaises(RemoteJobError) as caught:
            await asyncio.wait_for(job, 2)
        self.assertEqual(caught.exception.job_id, job.id)
        self.assertEqual(caught.exception.remote_error["type"], "ValueError")
        self.assertIn("deliberate test failure", caught.exception.remote_error["traceback"])

    async def test_cancel_wait_preserves_other_waiter_and_server(self):
        job = await self.start("slow", 0.3)
        other = asyncio.ensure_future(job)
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(job, 0.02)
        self.assertEqual((await self.record(job))["state"], "RUNNING")
        self.assertEqual(await asyncio.wait_for(other, 2), 0.3)
        self.assertEqual(await job, 0.3)

    async def test_remote_cancellation_waits_for_cleanup(self):
        job = await self.start("cleanup", 10)
        await self.wait_state(job, "RUNNING")
        self.assertEqual(await job.cancel_remote(), "CANCELLING")
        self.assertEqual(await job.cancel_remote(), "CANCELLING")
        self.assertFalse((await self.proxy.read_attribute("cleaned_up")).value)
        with self.assertRaises(RemoteJobCancelled):
            await asyncio.wait_for(job, 2)
        self.assertTrue((await self.proxy.read_attribute("cleaned_up")).value)
        self.assertEqual(await job.cancel_remote(), "CANCELLED")

    async def test_executor_cancellation_does_not_claim_thread_stopped(self):
        job = await self.start("blocking", 0.4)
        await self.wait_state(job, "RUNNING")
        self.assertEqual(await job.cancel_remote(), "CANCELLING")
        self.assertEqual(await self.proxy.echo(), "responsive")
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(job, 0.05)
        with self.assertRaises(RemoteJobCancelled):
            await asyncio.wait_for(job, 2)

    async def test_suppressed_cancellation_records_actual_result(self):
        job = await self.start("suppress_cancel", 10)
        await self.wait_state(job, "RUNNING")
        await job.cancel_remote()
        self.assertEqual(await asyncio.wait_for(job, 2), "completed despite cancellation")
        self.assertEqual((await self.record(job))["state"], "SUCCEEDED")

    async def test_unserializable_result_is_a_job_failure(self):
        job = await self.start("bad_result")
        with self.assertRaises(RemoteJobError) as caught:
            await asyncio.wait_for(job, 2)
        self.assertEqual(caught.exception.remote_error["type"], "TypeError")

    async def test_new_client_attaches_to_running_and_finished_job(self):
        job = await self.start("slow", 0.2)
        await job.close()
        with self.assertRaises(JobConnectionError):
            await job
        attached = await self.attach(job.id, proxy=self.make_proxy())
        self.assertEqual(await asyncio.wait_for(attached, 2), 0.2)
        finished = await self.attach(job.id, proxy=self.make_proxy())
        self.assertEqual(await finished, 0.2)
        self.assertIsNone(finished._subscription)

    async def test_missing_completion_event_is_reconciled(self):
        await self.proxy.emit_events(False)
        job = await self.start("slow", 0.15)
        self.assertEqual(await asyncio.wait_for(job, 2), 0.15)

    async def test_lost_start_reply_can_be_retried_without_duplicate_work(self):
        faulty = FaultyConnection(self.proxy)
        faulty.lose_start_reply = True
        with self.assertRaises(JobConnectionError) as caught:
            await self.start("slow", 0.2, proxy=faulty)
        job = await self.start("slow", 0.2, job_id=caught.exception.job_id)
        self.assertEqual(await asyncio.wait_for(job, 2), 0.2)
        again = await self.start("slow", 0.2, job_id=job.id)
        self.assertEqual(await again, 0.2)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 1)
        with self.assertRaises(tango.DevFailed):
            await self.start("slow", 0.3, job_id=job.id)

    async def test_transient_connection_failure_recovers(self):
        faulty = FaultyConnection(self.proxy)
        job = await self.start("slow", 0.25, proxy=faulty)
        faulty.offline = True
        job._wake.set()
        await asyncio.sleep(0.08)
        faulty.offline = False
        self.assertEqual(await asyncio.wait_for(job, 2), 0.25)

    async def test_connection_deadline_does_not_cancel_remote_job(self):
        faulty = FaultyConnection(self.proxy)
        job = await self.start("slow", 1, proxy=faulty)
        faulty.offline = True
        job._wake.set()
        with self.assertRaises(JobConnectionError) as caught:
            await asyncio.wait_for(job, 2)
        self.assertEqual(caught.exception.job_id, job.id)
        await job._monitor
        self.assertIsNone(job._subscription)
        self.assertEqual((await self.record(job))["state"], "RUNNING")
        faulty.offline = False
        attached = await self.attach(job.id)
        self.assertEqual(await asyncio.wait_for(attached, 2), 1)

    async def test_reinitialization_forgets_jobs_and_allows_resubmission(self):
        job = await self.start("slow", 0.2)
        await self.proxy.Init()
        # An existing observer may have seen cancellation just before Init
        # cleared history. A fresh attachment must see the ID as unavailable.
        with self.assertRaises((RemoteJobCancelled, JobUnavailable)):
            await asyncio.wait_for(job, 2)
        attached = await self.attach(job.id)
        with self.assertRaises(JobUnavailable):
            await attached
        # With no retained record, the same ID is accepted as a new submission.
        restarted = await self.start("slow", 0.2, job_id=job.id)
        self.assertEqual(await restarted, 0.2)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 1)

    async def test_unknown_id(self):
        job = await self.attach(str(uuid.uuid4()))
        with self.assertRaises(JobUnavailable):
            await job

    async def test_admission_checks(self):
        for name, args in [("echo", ()), ("slow", ("bad",)),
                           ("fail", ("unexpected",))]:
            with self.subTest(name=name):
                with self.assertRaises(tango.DevFailed):
                    await self.start(name, *args)
        job = await self.start("slow", 0.2)
        with self.assertRaises(tango.DevFailed):
            await self.start("immediate", "busy")
        self.assertEqual(await asyncio.wait_for(job, 2), 0.2)

    async def test_completed_history_is_bounded(self):
        first = await self.start("immediate", "first")
        await first
        retained = []
        for _ in range(JobTestDevice.history_limit):
            job = await self.start("immediate", "next")
            await job
            retained.append(job)
        self.assertEqual((await self.record(first))["state"], "UNAVAILABLE")
        for job in retained:
            record = await self.record(job)
            self.assertEqual(record["state"], "SUCCEEDED")
            self.assertFalse({"submitted", "started", "finished"} & record.keys())
