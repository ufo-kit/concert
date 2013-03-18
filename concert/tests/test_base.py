import unittest
import logbook
import time
from concert.base import ConcertObject
from concert.tests import VisitChecker


SLEEP_TIME = 0.005


class TestConcertObject(unittest.TestCase):
    def setUp(self):
        self.obj = ConcertObject()
        self.checker = VisitChecker()
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_subscribe(self):
        self.obj.subscribe('foo', self.checker.visit)
        self.obj.send('foo')
        time.sleep(SLEEP_TIME)
        self.assertTrue(self.checker.visited)

    def test_unsubscribe(self):
        self.obj.subscribe('foo', self.checker.visit)
        self.obj.unsubscribe('foo', self.checker.visit)
        self.obj.send('foo')
        time.sleep(SLEEP_TIME)
        self.assertFalse(self.checker.visited)
