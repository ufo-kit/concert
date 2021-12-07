import asyncio
import logging
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
    """Accumulate items in a list or a numpy array if *shape* is given, *dtype* is the data type.
    """

    def __init__(self, shape=None, dtype=None):
        import numpy as np

        self.items = [] if shape is None else np.empty(shape, dtype=dtype)

    @background
    def __call__(self, producer):
        """
        __call__(self, producer)

        Coroutine interface for processing in a pipeline.
        """
        if isinstance(self.items, list):
            return self._process(producer)
        else:
            return self._process_numpy(producer)

    async def _process(self, producer):
        """Stack data into a list."""
        # Clear results from possible previous execution but keep the list in the same place
        del self.items[:]

        async for item in producer:
            self.items.append(item)

    async def _process_numpy(self, producer):
        """Stack data into a numpy array."""
        i = 0

        async for item in producer:
            self.items[i] = item
            i += 1
