import unittest
import time
from concert.events.dispatcher import Dispatcher

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
        a1 = 0
        a2 = 1
        self.visited = 0
        
        def callback(sender):
            self.visited += 1
            
        self.dispatcher.subscribe([(None, 'foo')], callback)
        self.dispatcher.send(a1, 'foo')
        self.dispatcher.send(a2, 'foo')
        time.sleep(SLEEP_TIME)
        self.assertEqual(self.visited, 2, "{0} != {1}".format(self.visited, 2))