import logbook
from unittest import TestCase


def suppressed_logging(func):
    """Decorator for test functions."""
    def test_wrapper(*args, **kwargs):
        handler = logbook.NullHandler()
        handler.push_application()

        func(*args, **kwargs)

        handler.pop_application()

    return test_wrapper


class ConcertTest(TestCase):

    def setUp(self):
        self.handler = logbook.NullHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()
