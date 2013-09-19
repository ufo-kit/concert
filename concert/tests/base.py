import logbook
from unittest import TestCase


def suppressed_logging(func):
    """Decorator for test functions."""
    def test_wrapper(*args, **kwargs):
        handler = suppress_logging()

        func(*args, **kwargs)

        handler.pop_application()

    test_wrapper.__name__ = func.__name__

    return test_wrapper


class ConcertTest(TestCase):

    """Base class for tests which suppress logger output."""

    _multiprocess_can_split_ = False

    def setUp(self):
        self.handler = suppress_logging()

    def tearDown(self):
        self.handler.pop_application()


def suppress_logging():
    """Discard logger output and disable bubbling."""
    handler = logbook.NullHandler()
    handler.push_application()

    return handler
