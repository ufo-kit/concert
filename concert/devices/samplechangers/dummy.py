"""Dummy sample exchanger implementation."""

from concert.base import Selection
from concert.devices.samplechangers import base


class SampleChanger(base.SampleChanger):

    sample = Selection([None, 1, 2])

    async def __ainit__(self):
        await super(SampleChanger, self).__ainit__()
        self._sample = None

    async def _set_sample(self, sample):
        """Insert the sample in the holder."""
        self._sample = sample

    async def _get_sample(self):
        return self._sample
