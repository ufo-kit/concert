import unittest
from concert.events.dispatcher import Dispatcher
import time
from concert.devices.dummy import DummyDevice

SLEEP_TIME = 0.005


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = Dispatcher()

    def test_subscription(self):
        self.visited = False

        def callback(sender):
            self.visited = True

        self.dispatcher.subscribe(self, 'foo', callback)
        self.dispatcher.send(self, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertTrue(self.visited)

    def test_unsubscription(self):
        self.visited = False

        def callback(sender):
            self.visited = True

        self.dispatcher.subscribe(self, 'foo', callback)
        self.dispatcher.unsubscribe(self, 'foo', callback)
        self.dispatcher.send(self, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertFalse(self.visited)

    def test_wait(self):
        device_1 = DummyDevice()
        device_2 = DummyDevice()

        event_1 = device_1.set_value(15)
        event_2 = device_2.set_value(12)

        self.dispatcher.wait([event_1, event_2])
        self.assertEqual(device_1.get_value(), 15)
        self.assertEqual(device_2.get_value(), 12)

    def test_serialization(self):
        device = DummyDevice()

        event = device.set_value(1)
        self.dispatcher.wait([event])
        event = device.set_value(2)
        self.dispatcher.wait([event])
        event = device.set_value(3)
        self.dispatcher.wait([event])
        event = device.set_value(4)
        self.dispatcher.wait([event])

        self.assertEqual(4, device.get_value())
