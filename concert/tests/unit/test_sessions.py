"""Test sessions."""
import asyncio
from unittest import TestCase
from threading import Thread, Event
from concert.coroutines.base import start, run_in_loop_thread_blocking
from concert.session.utils import abort_awaiting
from concert.tests import slow, suppress_logging


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
