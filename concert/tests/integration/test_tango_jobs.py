"""Exercise the experimental protocol over real Tango RPC and change events."""
import asyncio
import json
import time
import unittest
import uuid

import pytest

tango = pytest.importorskip("tango")
from tango.asyncio_executor import AsyncioExecutor  # noqa: E402
from tango.test_context import DeviceTestContext  # noqa: E402

from concert.tests.util.tango_jobs import (  # noqa: E402
    JobConnectionError, JobDeviceProxy, JobTestDevice, JobUnavailable,
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
    """Count subscriptions and control the initial snapshot read."""

    def __init__(self, proxy):
        super().__init__(proxy)
        self.subscriptions = 0
        self.unsubscriptions = 0
        self.subscribed = asyncio.Event()
        self.release_snapshot = asyncio.Event()
        self.release_snapshot.set()
        self.fail_snapshot = False
        self.loop = asyncio.get_running_loop()

    def subscribe_event(self, *args, **kwargs):
        subscription = self.proxy.subscribe_event(*args, **kwargs)
        self.subscriptions += 1
        self.loop.call_soon_threadsafe(self.subscribed.set)
        return subscription

    async def unsubscribe_event(self, subscription):
        await self.proxy.unsubscribe_event(subscription)
        self.unsubscriptions += 1

    async def read_attribute(self, name):
        if name == "jobs":
            await self.release_snapshot.wait()
            if self.fail_snapshot:
                tango.Except.throw_exception("API_CommunicationFailed", "setup outage", "test")
        return await self.proxy.read_attribute(name)


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
        self.client = JobDeviceProxy(self.proxy, reconcile_interval=0.05, recovery_timeout=0.15)

    def make_proxy(self):
        proxy = tango.DeviceProxy(
            self.context.get_device_access(), green_mode=tango.GreenMode.Asyncio,
            asyncio_executor=AsyncioExecutor(loop=asyncio.get_running_loop()),
        )
        proxy.set_timeout_millis(500)
        return proxy

    async def asyncTearDown(self):
        await self.client.close()
        await self.proxy.Init()

    async def record(self, job):
        return json.loads(await self.proxy.GetJob(job.id))

    async def wait_state(self, job, state):
        async def wait():
            while (await self.record(job))["state"] != state:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait(), 2)

    async def test_ordinary_calls_and_close_do_not_initialize_monitoring(self):
        connection = SetupConnection(self.proxy)
        self.client.proxy = connection
        self.assertEqual(await self.client.echo(), "responsive")
        await self.client.close()
        await self.client.close()
        self.assertEqual(connection.subscriptions, 0)
        self.assertEqual(connection.unsubscriptions, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.client.start("immediate", "value")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.client.attach("unknown")

    async def test_concurrent_first_calls_share_setup(self):
        connection = SetupConnection(self.proxy)
        self.client.proxy = connection
        job, unknown = await asyncio.gather(
            self.client.start("slow", 0.1), self.client.attach("unknown"),
        )
        self.assertEqual(await asyncio.wait_for(job, 2), 0.1)
        with self.assertRaises(JobUnavailable):
            await unknown
        self.assertEqual(connection.subscriptions, 1)
        await asyncio.gather(self.client.close(), self.client.close())
        self.assertEqual(connection.unsubscriptions, 1)

    async def test_failed_setup_cleans_up_and_can_be_retried(self):
        connection = SetupConnection(self.proxy)
        self.client.proxy = connection
        connection.fail_snapshot = True
        with self.assertRaises(tango.DevFailed):
            await self.client.start("immediate", "value")
        self.assertEqual(connection.unsubscriptions, 1)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 0)
        connection.fail_snapshot = False
        self.assertEqual(await (await self.client.start("immediate", "retry")), "retry")
        self.assertEqual(connection.subscriptions, 2)
        await self.client.close()
        self.assertEqual(connection.unsubscriptions, 2)

    async def test_cancelled_first_caller_does_not_cancel_shared_setup(self):
        connection = SetupConnection(self.proxy)
        self.client.proxy = connection
        connection.release_snapshot.clear()
        first = asyncio.create_task(self.client.start("immediate", "cancelled"))
        try:
            await asyncio.wait_for(connection.subscribed.wait(), 2)
            second = asyncio.create_task(self.client.start("immediate", "survivor"))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
        finally:
            connection.release_snapshot.set()
        job = await asyncio.wait_for(second, 2)
        self.assertEqual(await job, "survivor")
        self.assertEqual(connection.subscriptions, 1)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 1)

    async def test_close_during_setup_releases_subscription(self):
        connection = SetupConnection(self.proxy)
        self.client.proxy = connection
        connection.release_snapshot.clear()
        starting = asyncio.create_task(self.client.start("immediate", "closed"))
        try:
            await asyncio.wait_for(connection.subscribed.wait(), 2)
            closing = asyncio.create_task(self.client.close())
            await asyncio.sleep(0)
        finally:
            connection.release_snapshot.set()
        await asyncio.wait_for(closing, 2)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await starting
        self.assertEqual(connection.unsubscriptions, 1)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 0)

    async def test_short_rpc_timeout_and_responsive_device(self):
        job = await self.client.start("slow", 1.1)
        self.assertNotIn((await self.record(job))["state"], {"SUCCEEDED", "FAILED"})
        for _ in range(3):
            self.assertEqual(await self.client.echo(), "responsive")
            self.assertGreaterEqual((await self.proxy.read_attribute("calls")).value, 1)
        self.assertEqual(await asyncio.wait_for(job, 3), 1.1)
        self.assertEqual(self.proxy.get_timeout_millis(), 500)

    async def test_completion_uses_change_events(self):
        # A successful wait within 2 seconds cannot come from a 60-second poll.
        await self.client.close()
        self.client = JobDeviceProxy(self.proxy, reconcile_interval=60)
        job = await self.client.start("slow", 0.2)
        self.assertEqual(await asyncio.wait_for(job, 2), 0.2)

    async def test_state_updates_while_interactive_event_loop_is_idle(self):
        job = await self.client.start("slow", 0.2)
        # Terminal IPython stops its loop between async cells. Deliberately
        # block this loop to reproduce that, leaving Tango's threads running.
        time.sleep(0.6)
        self.assertEqual(job.state, "SUCCEEDED")
        self.assertEqual(await job, 0.2)

    async def test_cancelled_state_updates_while_loop_is_idle(self):
        job = await self.client.start("cleanup", 10)
        await self.wait_state(job, "RUNNING")
        await job.cancel_remote()
        time.sleep(0.6)
        self.assertEqual(job.state, "CANCELLED")
        with self.assertRaises(RemoteJobCancelled):
            await job

    async def test_terminal_event_does_not_settle_with_stale_rpc_result(self):
        job = await self.client.start("slow", 0.3)
        self.client._monitor.cancel()
        await asyncio.gather(self.client._monitor, return_exceptions=True)
        running_record = await self.proxy.GetJob(job.id)
        self.assertEqual(json.loads(running_record)["state"], "RUNNING")
        time.sleep(0.6)
        self.assertEqual(job.state, "SUCCEEDED")

        class StaleRead(FaultyConnection):
            async def GetJob(self, job_id):
                return running_record

        self.client.proxy = StaleRead(self.proxy)
        await self.client._refresh(job)
        self.assertEqual(job.state, "SUCCEEDED")
        self.assertFalse(job._outcome.done())
        self.client.proxy = self.proxy
        await self.client._refresh(job)
        self.assertEqual(await job, 0.3)

    async def test_immediate_completion_and_legacy_command(self):
        self.assertEqual(await self.proxy.immediate("legacy"), "legacy")
        for _ in range(5):
            job = await self.client.start("immediate", "result")
            self.assertEqual(await asyncio.wait_for(job, 2), "result")

    async def test_concurrent_completion_reads_settle_handle_once(self):
        finished = await self.client.start("immediate", "result")
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

        self.client.proxy = SimultaneousReads(self.proxy)
        job = RemoteJob(self.client, finished.id)
        await asyncio.wait_for(asyncio.gather(
            self.client._refresh(job), self.client._refresh(job),
        ), 2)
        self.assertEqual(await job, "result")

    async def test_remote_failure(self):
        job = await self.client.start("fail")
        with self.assertRaises(RemoteJobError) as caught:
            await asyncio.wait_for(job, 2)
        self.assertEqual(caught.exception.job_id, job.id)
        self.assertEqual(caught.exception.remote_error["type"], "ValueError")
        self.assertIn("deliberate test failure", caught.exception.remote_error["traceback"])

    async def test_cancel_wait_preserves_other_waiter_and_server(self):
        job = await self.client.start("slow", 0.3)
        other = asyncio.ensure_future(job)
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(job, 0.02)
        self.assertEqual((await self.record(job))["state"], "RUNNING")
        self.assertEqual(await asyncio.wait_for(other, 2), 0.3)
        self.assertEqual(await job, 0.3)

    async def test_remote_cancellation_waits_for_cleanup(self):
        job = await self.client.start("cleanup", 10)
        await self.wait_state(job, "RUNNING")
        self.assertEqual(await job.cancel_remote(), "CANCELLING")
        self.assertEqual(await job.cancel_remote(), "CANCELLING")
        self.assertFalse((await self.proxy.read_attribute("cleaned_up")).value)
        with self.assertRaises(RemoteJobCancelled):
            await asyncio.wait_for(job, 2)
        self.assertTrue((await self.proxy.read_attribute("cleaned_up")).value)
        self.assertEqual(await job.cancel_remote(), "CANCELLED")

    async def test_executor_cancellation_does_not_claim_thread_stopped(self):
        job = await self.client.start("blocking", 0.4)
        await self.wait_state(job, "RUNNING")
        self.assertEqual(await job.cancel_remote(), "CANCELLING")
        self.assertEqual(await self.client.echo(), "responsive")
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(job, 0.05)
        with self.assertRaises(RemoteJobCancelled):
            await asyncio.wait_for(job, 2)

    async def test_suppressed_cancellation_records_actual_result(self):
        job = await self.client.start("suppress_cancel", 10)
        await self.wait_state(job, "RUNNING")
        await job.cancel_remote()
        self.assertEqual(await asyncio.wait_for(job, 2), "completed despite cancellation")
        self.assertEqual((await self.record(job))["state"], "SUCCEEDED")

    async def test_unserializable_result_is_a_job_failure(self):
        job = await self.client.start("bad_result")
        with self.assertRaises(RemoteJobError) as caught:
            await asyncio.wait_for(job, 2)
        self.assertEqual(caught.exception.remote_error["type"], "TypeError")

    async def test_new_client_attaches_to_running_and_finished_job(self):
        job = await self.client.start("slow", 0.2)
        await self.client.close()
        with self.assertRaises(JobConnectionError):
            await job
        second = JobDeviceProxy(self.make_proxy(), reconcile_interval=0.05)
        try:
            attached = await second.attach(job.id)
            self.assertEqual(await asyncio.wait_for(attached, 2), 0.2)
        finally:
            await second.close()
        third = JobDeviceProxy(self.make_proxy())
        try:
            self.assertEqual(await (await third.attach(job.id)), 0.2)
        finally:
            await third.close()

    async def test_missing_completion_event_is_reconciled(self):
        await self.proxy.emit_events(False)
        job = await self.client.start("slow", 0.15)
        self.assertEqual(await asyncio.wait_for(job, 2), 0.15)

    async def test_lost_start_reply_can_be_retried_without_duplicate_work(self):
        faulty = FaultyConnection(self.proxy)
        faulty.lose_start_reply = True
        self.client.proxy = faulty
        with self.assertRaises(JobConnectionError) as caught:
            await self.client.start("slow", 0.2)
        job = await self.client.start("slow", 0.2, job_id=caught.exception.job_id)
        self.assertEqual(await asyncio.wait_for(job, 2), 0.2)
        again = await self.client.start("slow", 0.2, job_id=job.id)
        self.assertEqual(await again, 0.2)
        self.assertEqual((await self.proxy.read_attribute("calls")).value, 1)
        with self.assertRaises(tango.DevFailed):
            await self.client.start("slow", 0.3, job_id=job.id)

    async def test_transient_connection_failure_recovers(self):
        faulty = FaultyConnection(self.proxy)
        self.client.proxy = faulty
        job = await self.client.start("slow", 0.25)
        faulty.offline = True
        self.client._wake.set()
        await asyncio.sleep(0.08)
        faulty.offline = False
        self.assertEqual(await asyncio.wait_for(job, 2), 0.25)

    async def test_connection_deadline_does_not_cancel_remote_job(self):
        faulty = FaultyConnection(self.proxy)
        self.client.proxy = faulty
        job = await self.client.start("slow", 1)
        faulty.offline = True
        self.client._wake.set()
        with self.assertRaises(JobConnectionError) as caught:
            await asyncio.wait_for(job, 2)
        self.assertEqual(caught.exception.job_id, job.id)
        self.assertEqual((await self.record(job))["state"], "RUNNING")
        faulty.offline = False
        attached = await self.client.attach(job.id)
        self.assertEqual(await asyncio.wait_for(attached, 2), 1)

    async def test_reinitialization_invalidates_old_ids(self):
        job = await self.client.start("slow", 10)
        await self.proxy.Init()
        # An existing observer may have seen cancellation just before Init
        # cleared history. A fresh attachment must see the ID as unavailable.
        with self.assertRaises((RemoteJobCancelled, JobUnavailable)):
            await asyncio.wait_for(job, 2)
        attached = await self.client.attach(job.id)
        with self.assertRaises(JobUnavailable):
            await attached
        with self.assertRaises(tango.DevFailed):
            await self.client.start("slow", 0, job_id=job.id)

    async def test_unknown_id(self):
        snapshot = json.loads((await self.proxy.read_attribute("jobs")).value)
        job = await self.client.attach(f"{snapshot['instance']}/{uuid.uuid4()}")
        with self.assertRaises(JobUnavailable):
            await job

    async def test_admission_checks(self):
        for name, args in [("echo", ()), ("forbidden", ()), ("slow", ("bad",)),
                           ("fail", ("unexpected",))]:
            with self.subTest(name=name):
                with self.assertRaises(tango.DevFailed):
                    await self.client.start(name, *args)
        job = await self.client.start("slow", 0.2)
        with self.assertRaises(tango.DevFailed):
            await self.client.start("immediate", "busy")
        self.assertEqual(await asyncio.wait_for(job, 2), 0.2)

    async def test_completed_history_is_bounded(self):
        first = await self.client.start("immediate", "first")
        await first
        for _ in range(JobTestDevice.history_limit):
            await (await self.client.start("immediate", "next"))
        self.assertEqual((await self.record(first))["state"], "UNAVAILABLE")
        snapshot = json.loads((await self.proxy.read_attribute("jobs")).value)
        self.assertEqual(len(snapshot["jobs"]), JobTestDevice.history_limit)
