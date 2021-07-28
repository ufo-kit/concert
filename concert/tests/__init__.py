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
    """Discard unit on x and y and assert that they are almost equal."""
    try:
        x[0]
    except Exception:
        x = [x]

    try:
        y[0]
    except Exception:
        y = [y]

    assert len(x) == len(y)

    for i in range(len(x)):
        diff = np.abs(x[i] - y[i]).to_base_units().magnitude
        assert diff <= epsilon, f"x != y at i={i}: {x[i]} != {y[i]} (within {epsilon})"


class TestCase(unittest.IsolatedAsyncioTestCase):

    """Base class for tests which suppress logger output."""

    def setUp(self):
        suppress_logging()
        cfg.PROGRESS_BAR = False
