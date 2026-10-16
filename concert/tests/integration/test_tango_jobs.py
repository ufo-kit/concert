"""Exercise the experimental protocol over real Tango RPC and change events."""
import asyncio
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

tango = pytest.importorskip("tango")
from tango.asyncio_executor import AsyncioExecutor  # noqa: E402
from tango.test_context import MultiDeviceTestContext  # noqa: E402

from concert.base import AsyncObject  # noqa: E402
from concert.session.utils import abort_awaiting  # noqa: E402
from concert.tests.util.tango_jobs import (  # noqa: E402
    JobConnectionError, JobDevice, JobTestDevice, JobUnavailable,
    RemoteJob, RemoteJobCancelled, RemoteJobError, job_command,
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
        self.attributes = []
        self.events = []
        self.callback_loops = []
        self.unsubscriptions = 0
        self.subscribed = asyncio.Event()
        self.loop = asyncio.get_running_loop()
        self.release_subscription = asyncio.Event()
        self.release_subscription.set()

    async def subscribe_event(self, *args, **kwargs):
        attr_name, event_type, callback = args

        def record_event(event):
            self.callback_loops.append(asyncio.get_running_loop())
            self.events.append(event)
            callback(event)

        subscription = await self.proxy.subscribe_event(
            attr_name, event_type, record_event, **kwargs)
        self.subscriptions += 1
        self.attributes.append(attr_name)
        self.subscribed.set()
        await self.release_subscription.wait()
        return subscription

    async def unsubscribe_event(self, subscription):
        await self.proxy.unsubscribe_event(subscription)
        self.unsubscriptions += 1


class TestJobAttributeNames(unittest.TestCase):
    def test_default_and_custom_names(self):
        async def operation(self):
            pass

        self.assertEqual(job_command()(operation)._job_spec[2], "operation_job_state")
        self.assertEqual(job_command(attr_name="custom")(operation)._job_spec[2], "custom")
        for name in ("", 12):
            with self.assertRaises(ValueError):
                job_command(attr_name=name)

    def test_collisions_are_rejected_before_creating_attributes(self):
        for names in (("shared", "SHARED"), ("calls",), ("STATE",), ("Status",)):
            with self.subTest(names=names):
                device = SimpleNamespace(
                    _operations={str(i): (None, None, name) for i, name in enumerate(names)},
                    get_device_attr=lambda: SimpleNamespace(
                        get_attr_nb=lambda: 3,
                        get_attr_by_ind=lambda i: SimpleNamespace(
                            get_name=lambda: ("calls", "State", "Status")[i])),
                    add_attribute=Mock(),
                )
                with self.assertRaisesRegex(ValueError, "collision"):
                    JobDevice.initialize_dynamic_attributes(device)
                device.add_attribute.assert_not_called()


class TestTangoJobs(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # A separate process avoids mixing Tango's server and client executors.
        cls.context = MultiDeviceTestContext([{
            "class": JobTestDevice,
            "devices": [{"name": "test/jobs/first"}, {"name": "test/jobs/second"}],
        }], process=True, host="127.0.0.1")
        cls.context.start()

    @classmethod
    def tearDownClass(cls):
        cls.context.stop()

    async def asyncSetUp(self):
        self.proxy = self.make_proxy()
        await self.proxy.Init()

    def make_proxy(self, name="test/jobs/first"):
        proxy = tango.DeviceProxy(
            self.context.get_device_access(name), green_mode=tango.GreenMode.Asyncio,
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
        if args:
            options["arg"], = args
        job = await RemoteJob(proxy=self.proxy if proxy is None else proxy, command=name, **options)
        self.addAsyncCleanup(job.close)
        return job

    async def attach(self, job_id, proxy=None, **options):
        options.setdefault("reconcile_interval", 0.05)
        options.setdefault("recovery_timeout", 0.15)
        job = await RemoteJob(proxy=self.proxy if proxy is None else proxy,
                              job_id=job_id, **options)
        self.addAsyncCleanup(job.close)
        return job

    async def test_async_constructor_initializes_one_job(self):
        job = await self.start("slow", 0.1)
        self.assertIsInstance(job, AsyncObject)
        self.assertIsInstance(job, RemoteJob)
        self.assertFalse(hasattr(RemoteJob, "start"))
        self.assertFalse(hasattr(RemoteJob, "attach"))
        self.assertEqual(await job, 0.1)

    async def test_constructor_requires_command_or_id(self):
        for options in ({}, {"arg": 1}, {"job_id": str(uuid.uuid4()), "arg": 1}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    await RemoteJob(proxy=self.proxy, **options)
        # Explicit None is an argument, not omission of a void command's input.
        with self.assertRaises(tango.DevFailed):
            await RemoteJob(proxy=self.proxy, command="fail", arg=None)

    async def test_command_attributes_and_custom_name_discovery(self):
        attributes = list(self.proxy.get_attribute_list())
        self.assertIn("immediate_job_state", attributes)
        self.assertIn("delay_state", attributes)
        self.assertNotIn("slow_job_state", attributes)
        self.assertNotIn("job_changed", attributes)
        connection = SetupConnection(self.proxy)
        job = await self.start("slow", 0.2, proxy=connection)
        self.assertEqual((await self.record(job))["attr_name"], "delay_state")
        attached = await self.attach(job.id, proxy=connection)
        self.assertEqual(connection.attributes, ["delay_state", "delay_state"])
        self.assertEqual(await attached, 0.2)
        self.assertEqual(await job, 0.2)
        value = json.loads((await self.proxy.read_attribute("delay_state")).value)
        self.assertEqual(value, {"id": job.id, "state": "SUCCEEDED"})

    async def test_other_commands_do_not_deliver_events_to_handle(self):
        connection = SetupConnection(self.proxy)
        job = await self.start("slow", 0.1, proxy=connection)
        # Keep this subscription alive after server completion to test routing.
        job._refresh = AsyncMock()
        await self.wait_state(job, "SUCCEEDED")
        other = await self.start("immediate", "other command")
        await other
        await asyncio.sleep(0.1)
        ids = {json.loads(event.attr_value.value)["id"] for event in connection.events
               if not event.err and event.attr_value is not None}
        self.assertIn(job.id, ids)
        self.assertNotIn(other.id, ids)
        await job.close()

    async def test_finished_and_unknown_jobs_do_not_subscribe(self):
        job = await self.start("immediate", "finished")
        await job
        connection = SetupConnection(self.proxy)
        finished = await self.attach(job.id, proxy=connection)
        self.assertEqual(await finished, "finished")
        unknown = await self.attach(str(uuid.uuid4()), proxy=connection)
        with self.assertRaises(JobUnavailable):
            await unknown
        self.assertEqual(connection.subscriptions, 0)

    async def test_completion_between_discovery_and_subscription(self):
        class DelayedDiscovery(SetupConnection):
            reads = 0

            async def GetJob(self, job_id):
                record = await super().GetJob(job_id)
                self.reads += 1
                if self.reads == 1:
                    self.initial_state = json.loads(record)["state"]
                    await asyncio.sleep(0.3)
                return record

        await self.proxy.emit_events(False)
        connection = DelayedDiscovery(self.proxy)
        job = await self.start("slow", 0.15, proxy=connection, reconcile_interval=60)
        self.assertEqual(connection.initial_state, "RUNNING")
        self.assertEqual(await asyncio.wait_for(job, 2), 0.15)
        self.assertEqual(connection.reads, 2)
        self.assertEqual(connection.unsubscriptions, 1)

    async def test_initial_discovery_failure_recovers(self):
        connection = SetupConnection(self.proxy)
        connection.offline = True
        job = await self.start("slow", 0.4, proxy=connection)
        self.assertEqual(connection.subscriptions, 0)
        await asyncio.sleep(0.05)
        connection.offline = False
        self.assertEqual(await asyncio.wait_for(job, 2), 0.4)
        self.assertEqual(connection.attributes, ["delay_state"])

    async def test_dynamic_attributes_survive_init_and_are_device_local(self):
        other = self.make_proxy("test/jobs/second")
        await other.Init()
        attributes = list(self.proxy.get_attribute_list())
        for _ in range(2):
            await self.proxy.Init()
            self.assertEqual(list(self.proxy.get_attribute_list()), attributes)
        first, second = await asyncio.gather(
            self.start("slow", 0.2), self.start("slow", 0.2, proxy=other),
        )
        self.assertEqual(await first, 0.2)
        self.assertEqual(await second, 0.2)
        for proxy, job in ((self.proxy, first), (other, second)):
            value = json.loads((await proxy.read_attribute("delay_state")).value)
            self.assertEqual(value, {"id": job.id, "state": "SUCCEEDED"})

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
        self.assertTrue(connection.callback_loops)
        self.assertTrue(all(loop is asyncio.get_running_loop()
                            for loop in connection.callback_loops))

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

        job = await self.start("slow", 1, proxy=FailedSubscription(self.proxy))
        with self.assertRaises(JobConnectionError) as caught:
            await job
        job = await self.attach(caught.exception.job_id)
        self.assertEqual(await job, 1)
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
        change = json.loads((await self.proxy.read_attribute("delay_state")).value)
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
        job._refresh = AsyncMock()  # Test event ordering before the result is fetched.
        running = json.loads((await self.proxy.read_attribute("delay_state")).value)
        self.assertEqual(set(running), {"id", "state"})
        self.assertEqual(running["id"], job.id)
        await asyncio.sleep(0.6)
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
        del job._refresh
        await job._refresh()
        self.assertEqual(await job, 0.3)
        self.assertEqual(job.state, "SUCCEEDED")

    async def test_state_updates_without_awaiting_job(self):
        job = await self.start("slow", 0.2)
        # Concert's asyncio-enabled prompt lets tasks run between user commands.
        await asyncio.sleep(0.6)
        self.assertEqual(job.state, "SUCCEEDED")
        self.assertEqual(await job, 0.2)

    async def test_cancelled_state_updates_without_awaiting_job(self):
        job = await self.start("cleanup", 10)
        await self.wait_state(job, "RUNNING")
        await job.cancel_remote()
        await asyncio.sleep(0.6)
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
        await asyncio.sleep(0.6)
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
        job = await self.start("slow", 0.1)
        job._refresh = AsyncMock()  # Keep the outcome pending until the two test reads.
        await self.wait_state(job, "SUCCEEDED")
        response = await self.proxy.GetJob(job.id)
        both_reading = asyncio.Event()

        class SimultaneousReads(FaultyConnection):
            reads = 0

            async def GetJob(self, job_id):
                self.reads += 1
                if self.reads == 2:
                    both_reading.set()
                await both_reading.wait()
                return response

        job.proxy = SimultaneousReads(self.proxy)
        del job._refresh
        await asyncio.wait_for(asyncio.gather(
            job._refresh(), job._refresh(),
        ), 2)
        self.assertEqual(await job, 0.1)

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

    async def test_concert_abort_stops_monitoring_but_not_server_job(self):
        for immediately in (True, False):
            with self.subTest(immediately=immediately):
                job = await self.start("slow", 10)
                if not immediately:
                    await asyncio.sleep(0)  # Let the watcher enter its try/finally.
                self.assertTrue(job._monitor._is_concert_task)
                abort_awaiting()  # The action bound to Ctrl-K in Concert.
                with self.assertRaises(JobConnectionError):
                    await asyncio.wait_for(job, 2)

                async def wait_for_cleanup():
                    while job._subscription is not None:
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_for_cleanup(), 2)
                self.assertEqual((await self.record(job))["state"], "RUNNING")
                await self.proxy.Init()

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
