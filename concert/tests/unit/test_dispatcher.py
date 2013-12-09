import time
from concert.tests import VisitChecker, TestCase
from concert.async import Dispatcher

SLEEP_TIME = 0.05


class TestDispatcher(TestCase):

    def setUp(self):
        super(TestDispatcher, self).setUp()
        self.dispatcher = Dispatcher()
        self.checker = VisitChecker()

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
