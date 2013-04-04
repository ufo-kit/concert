'''
Created on Apr 4, 2013

@author: farago
'''
from concert.devices.dummy import DummyDevice
import unittest
from concurrent.futures import Future
import time
from concert.asynchronous import async, wait
from concert.tests import slow
import random
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

    @slow
    def test_wait(self):
        @async
        def long_func():
            time.sleep(random.random()/50.)

        futs = []
        for i in range(10):
            futs.append(long_func())

        wait(futs)

        for future in futs:
            self.assertTrue(future.done(), "Not all futures finished.")

        # Test error raising for wrong arguments.
        self.assertRaises(Exception, wait, [func(0)])

        # test error raising for exception inside an asynchronous function.
        self.assertRaises(Exception, wait, [bad_func()])

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

        # Test error raising for wrong arguments.
        self.assertRaises(Exception, func(0).wait)

        # test error raising for exception inside an asynchronous function.
        self.assertRaises(Exception, bad_func().wait)

    def test_async_function(self):
        future = func()
        self.assertIsInstance(future, Future,
                              "Function was not run asynchronously.")

    def test_async_accessors(self):
        future1 = self.device.set_value(15)
        future2 = self.device.get_value()

        self.assertIsInstance(future1, Future,
                              "Setter accessor does not return a future.")
        self.assertIsInstance(future2, Future,
                              "Getter accessor does not return a future.")

    def test_async_parameter(self):
        future1 = self.device["value"].set(15)
        future2 = self.device["value"].get()

        self.assertIsInstance(future1, Future,
                              "Setter does not return a future.")
        self.assertIsInstance(future2, Future,
                              "Getter does not return a future.")

    def test_async_method(self):
        future = self.device.do_nothing()

        self.assertIsInstance(future, Future,
                              "Asynchronous method does not return a future.")
