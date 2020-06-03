import concert.config as cfg
import logging
import unittest
import numpy as np


def slow(func):
    """Mark a test method as slow running.

    You can skip these test cases with nose by running ``nosetest -a '!slow'``
    or calling ``make check-fast``.
    """
    func.slow = 1
    return func


def suppressed_logging(func):
    """Decorator for test functions."""
    def test_wrapper(*args, **kwargs):
        suppress_logging()
        func(*args, **kwargs)

    test_wrapper.__name__ = func.__name__
    return test_wrapper


def suppress_logging():
    """Discard logger output and disable bubbling."""
    logging.disable(logging.CRITICAL)


def assert_almost_equal(x, y, epsilon=1e-10):
    """Discard unit on x and y and assert that they are almost equal"""
    x = x.to_base_units().magnitude
    y = y.to_base_units().magnitude

    assert len(np.where(np.abs(x - y) > epsilon)[0]) == 0, "{} != {} (within {})".format(x, y,
                                                                                         epsilon)


class TestCase(unittest.TestCase):

    """Base class for tests which suppress logger output."""

    def setUp(self):
        suppress_logging()
        cfg.PROGRESS_BAR = False


class VisitChecker(object):

    """Use this to check that a callback was called."""

    def __init__(self):
        self.visited = False

    def visit(self, *args, **kwargs):
        self.visited = True
