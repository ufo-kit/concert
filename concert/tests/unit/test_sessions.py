"""Test sessions."""
import asyncio
from threading import Thread, Event
from concert.coroutines.base import start, run_in_loop_thread_blocking
from concert.quantities import q
from concert.session.utils import abort_awaiting, check_emergency_stop
from concert.tests import slow, suppress_logging, TestCase


async def corofunc(inst):
    inst.started.set()
    try:
        await asyncio.sleep(1)
        return 1
    except asyncio.CancelledError:
        inst.cancelled = True


@slow
class TestAborting(TestCase):

    def setUp(self):
        suppress_logging()
        self.cancelled = False
        self.started = Event()
        super().setUp()

    def test_abort_awaiting_background(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = start(corofunc(self))
        abort_awaiting(background=True)
        assert task.cancelled()

    def test_abort_awaiting_thread(self):
        result = None

        def run():
            nonlocal result

            result = run_in_loop_thread_blocking(corofunc(self))

        thread = Thread(target=run)
        thread.start()
        self.started.wait()
        abort_awaiting(background=True)
        thread.join()
        self.assertEqual(result, None)

    async def test_check_emergency_stop(self):
        class Callable:
            num_called = 0
            aborted = False

            def __call__(self):
                result = False
                if self.num_called == 2:
                    self.aborted = True
                    result = True

                self.num_called += 1

                return result

        clb = Callable()
        cancelled_task = start(corofunc(self))
        task = start(check_emergency_stop(clb, poll_interval=1 * q.ms))

        await asyncio.sleep(.1)

        await cancelled_task
        assert self.cancelled
        assert clb.aborted
        # This means callable has been called after the flag has been set, enabling it to wait for
        # clearing the flag
        assert clb.num_called > 3

        # After the flag has been cleared, no more aborting of newly created tasks
        running_task = start(corofunc(self))
        await asyncio.sleep(.1)
        await running_task
        assert running_task.result() == 1
