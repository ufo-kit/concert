import time
import random
from concert.devices.dummy import DummyDevice
from concert.async import async, wait, KillException, HAVE_GEVENT
from concert.tests import slow, TestCase


@async
def func():
    pass


@async
def bad_func():
    raise RuntimeError


class TestAsync(TestCase):

    def setUp(self):
        super(TestAsync, self).setUp()
        self.device = DummyDevice()

    @slow
    def test_wait(self):
        @async
        def long_func():
            time.sleep(random.random() / 50.)

        futs = []
        for i in range(10):
            futs.append(long_func())

        wait(futs)

        for future in futs:
            self.assertTrue(future.done(), "Not all futures finished.")

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            func(0).join()

        with self.assertRaises(RuntimeError):
            bad_func().join()

    def test_kill(self):
        d = {'killed': False}

        @async
        def long_op(d):
            try:
                time.sleep(10)
            except KillException:
                d['killed'] = True

        future = long_op(d)
        time.sleep(0)
        future.kill()

        if HAVE_GEVENT:
            self.assertTrue(d['killed'])
