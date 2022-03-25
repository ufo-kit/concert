import asyncio
import logging
import numpy as np
from concert.config import AIODEBUG
from concert.coroutines.base import background


LOG = logging.getLogger(__name__)


class Result(object):
    """
    The object is callable and when called it becomes a coroutine which accepts
    items and stores them in a variable which allows the user to obtain the
    last stored item at any time point.
    """
    def __init__(self):
        self.result = None

    @background
    async def __call__(self, producer):
        """
        __call__(self, producer)

        When the object is called we store every item in an attribute which
        can be recovered later.
        """
        async for item in producer:
            self.result = item


@background
async def null(producer):
    """
    null(producer)

    A black-hole.
    """
    try:
        async for item in producer:
            pass
    except asyncio.CancelledError:
        LOG.log(AIODEBUG, 'null cancelled')
        raise


class Accumulate(object):
    """Accumulate items in a list or a numpy array if *shape* is given, *dtype* is the data type. If
    *reset_on_call* is True, the saved values will be overwritten every time the accumulator is
    called, otherwise they will be appended.
    """

    def __init__(self, shape=None, dtype=None, reset_on_call=True):
        self._shape = shape
        self._dtype = dtype
        self.reset_on_call = reset_on_call
        self.items = [] if shape is None else np.empty((0,) + self._shape[1:], dtype=self._dtype)

    @background
    def __call__(self, producer):
        """
        __call__(self, producer)

        Coroutine interface for processing in a pipeline.
        """
        if self.reset_on_call:
            self.reset()

        if isinstance(self.items, list):
            return self._process(producer)
        else:
            return self._process_numpy(producer)

    def reset(self):
        if isinstance(self.items, list):
            del self.items[:]
        else:
            self.items = np.empty((0,) + self._shape[1:], dtype=self._dtype)

    async def _process(self, producer):
        """Stack data into a list."""
        async for item in producer:
            self.items.append(item)

    async def _process_numpy(self, producer):
        """Stack data into a numpy array."""
        if self.reset_on_call and len(self.items):
            current = self.items
        else:
            current = np.empty(self._shape, dtype=self._dtype)

        i = 0
        async for item in producer:
            current[i] = item
            i += 1

        if not self.reset_on_call:
            self.items = np.concatenate((self.items, current))
