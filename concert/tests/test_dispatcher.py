import unittest
import logbook
import time
from concert.tests import VisitChecker
from concert.base import wait
from concert.events.dispatcher import Dispatcher
from concert.devices.dummy import DummyDevice

SLEEP_TIME = 0.005


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = Dispatcher()
        self.checker = VisitChecker()
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_subscription(self):
        self.dispatcher.subscribe(self, 'foo', self.checker.visit)
        self.dispatcher.send(self, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertTrue(self.checker.visited)

    def test_unsubscription(self):
        self.dispatcher.subscribe(self, 'foo', self.checker.visit)
        self.dispatcher.unsubscribe(self, 'foo', self.checker.visit)
        self.dispatcher.send(self, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertFalse(self.checker.visited)

    def test_wait(self):
        device_1 = DummyDevice()
        device_2 = DummyDevice()

        event_1 = device_1.set_value(15)
        event_2 = device_2.set_value(12)

        wait([event_1, event_2])
        self.assertEqual(device_1.get_value(), 15)
        self.assertEqual(device_2.get_value(), 12)

    def test_serialization(self):
        device = DummyDevice()

        event = device.set_value(1)
        wait([event])
        event = device.set_value(2)
        wait([event])
        event = device.set_value(3)
        wait([event])
        event = device.set_value(4)
        wait([event])

        self.assertEqual(4, device.get_value())
