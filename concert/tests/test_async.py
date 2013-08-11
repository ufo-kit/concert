import unittest
import time
import random
import logbook
from concurrent.futures import Future
from testfixtures import ShouldRaise
from concert.devices.dummy import DummyDevice
from concert.asynchronous import async, wait
from concert.tests import slow
from concert import asynchronous


@async
def func():
    pass


@async
def bad_func():
    raise RuntimeError


class TestAsync(unittest.TestCase):

    def setUp(self):
        self.device = DummyDevice()
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

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
        with ShouldRaise(TypeError):
            wait([func(0)])

        with ShouldRaise(RuntimeError):
            wait([bad_func()])

    def test_is_async(self):
        self.assertTrue(asynchronous.is_async(func))

        def sync_func():
            pass

        self.assertFalse(asynchronous.is_async(sync_func))

    def test_futures(self):
        future1 = func()
        future2 = func().wait()
        self.assertEqual(future1.__class__, future2.__class__,
                         "Wait method does not return a future.")

        with ShouldRaise(TypeError):
            func(0).wait()

        with ShouldRaise(RuntimeError):
            bad_func().wait()

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
