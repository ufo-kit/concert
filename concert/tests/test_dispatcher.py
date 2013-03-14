import unittest
from concert.events.dispatcher import Dispatcher, dispatcher
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

        self.dispatcher.subscribe([(self, 'foo')], callback)
        self.dispatcher.send(self, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertTrue(self.visited)

    def test_unsubscription(self):
        self.visited = False

        def callback(sender):
            self.visited = True

        self.dispatcher.subscribe([(self, 'foo')], callback)
        self.dispatcher.unsubscribe([(self, 'foo')], callback)
        self.dispatcher.send(self, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertFalse(self.visited)

    def test_multiple_senders(self):
        a_1 = 0
        a_2 = 1
        self.visited = 0

        def callback(sender):
            self.visited += 1

        self.dispatcher.subscribe([(None, 'foo')], callback)
        self.dispatcher.send(a_1, 'foo')
        self.dispatcher.send(a_2, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertEqual(self.visited, 2, "{0} != {1}".format(self.visited, 2))

    def test_wait(self):
        d_1 = DummyDevice()
        d_2 = DummyDevice()

        e_1 = d_1.set_value(15)
        e_2 = d_2.set_value(12)

        dispatcher.wait([e_1, e_2])
        self.assertEqual(d_1.get_value(), 15)
        self.assertEqual(d_2.get_value(), 12)
