from .base import coroutine


class Result(object):
    """
    The object is callable and when called it becomes a coroutine which accepts
    items and stores them in a variable which allows the user to obtain the
    last stored item at any time point.
    """
    def __init__(self):
        self.result = None

    @coroutine
    def __call__(self):
        """
        __call__(self)

        When the object is called we store every item in an attribute which
        can be recovered later.
        """
        while True:
            self.result = yield


@coroutine
def null():
    """
    null()

    A black-hole.
    """
    while True:
        yield


class Accumulate(object):
    """Accumulate items in a list or a numpy array if *shape* is given, *dtype* is the data type.
    """

    def __init__(self, shape=None, dtype=None):
        import numpy as np

        self.items = [] if shape is None else np.empty(shape, dtype=dtype)

    def __call__(self):
        """
        __call__(self)

        Coroutine interface for processing in a pipeline.
        """
        if isinstance(self.items, list):
            return self._process()
        else:
            return self._process_numpy()

    @coroutine
    def _process(self):
        """Stack data into a list."""
        # Clear results from possible previous execution but keep the list in the same place
        del self.items[:]

        while True:
            item = yield
            self.items.append(item)

    @coroutine
    def _process_numpy(self):
        """Stack data into a numpy array."""
        i = 0

        while True:
            item = yield
            self.items[i] = item
            i += 1
