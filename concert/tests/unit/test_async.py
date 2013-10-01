import time
import random
from concurrent.futures import Future
from concert.devices.dummy import DummyDevice
from concert.helpers import async, wait, is_async
from concert.tests import slow
from concert.tests.base import ConcertTest


@async
def func():
    pass


@async
def bad_func():
    raise RuntimeError


class TestAsync(ConcertTest):

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
        self.assertRaises(TypeError, wait, [func(0)])
        self.assertRaises(RuntimeError, wait, [bad_func()])

    def test_is_async(self):
        self.assertTrue(is_async(func))

        def sync_func():
            pass

        self.assertFalse(is_async(sync_func))

    def test_futures(self):
        future1 = func()
        future2 = func().wait()
        self.assertEqual(future1.__class__, future2.__class__,
                         "Wait method does not return a future.")

        self.assertRaises(TypeError, func(0).wait)
        self.assertRaises(RuntimeError, bad_func().wait)

    def test_async_function(self):
        future = func()
        self.assertTrue(isinstance(future, Future),
                        "Function was not run asynchronously.")

    def test_async_accessors(self):
        future1 = self.device.set_value(15)
        future2 = self.device.get_value()

        self.assertTrue(isinstance(future1, Future),
                        "Setter accessor does not return a future.")
        self.assertTrue(isinstance(future2, Future),
                        "Getter accessor does not return a future.")

        wait([future1, future2])

    def test_async_parameter(self):
        future1 = self.device["value"].set(15)
        future2 = self.device["value"].get()

        self.assertTrue(isinstance(future1, Future),
                        "Setter does not return a future.")
        self.assertTrue(isinstance(future2, Future),
                        "Getter does not return a future.")

        wait([future1, future2])

    def test_async_method(self):
        future = self.device.do_nothing()

        self.assertTrue(isinstance(future, Future),
                        "Asynchronous method does not return a future.")
