def slow(func):
    """Mark a test method as slow running.

    You can skip these test cases with nose by running ``nosetest -a '!slow'``
    or calling ``make check-fast``.
    """
    func.slow = 1
    return func


class VisitChecker(object):

    """Use this to check that a callback was called."""

    def __init__(self):
        self.visited = False

    def visit(self, *args, **kwargs):
        self.visited = True


def assert_almost_equal(x, y, epsilon=1e-10):
    """Discard unit on x and y and assert that they are almost equal"""
    assert abs((x - y).magnitude) < epsilon
