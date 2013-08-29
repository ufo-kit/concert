import logbook
from unittest import TestCase


class ConcertTest(TestCase):

    def setUp(self):
        self.handler = logbook.NullHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()
