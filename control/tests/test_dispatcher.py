import unittest
import time
from control.events.dispatcher import Dispatcher


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = Dispatcher()

    def test_subscription(self):
        self.visited = False

        def callback(sender):
            self.visited = True

        self.dispatcher.subscribe(self, 'foo', callback)
        self.dispatcher.send(self, 'foo')
        time.sleep(0.005)
        self.assertTrue(self.visited)
