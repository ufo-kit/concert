"""Dummy sample exchanger implementation."""

from concert.base import Selection
from concert.devices.samplechangers import base


class SampleChanger(base.SampleChanger):

    sample = Selection([None, 1, 2])

    def __init__(self):
        super(SampleChanger, self).__init__()
        self._sample = None

    def _set_sample(self, sample):
        """Insert the sample in the holder."""
        self._sample = sample

    def _get_sample(self):
        return self._sample
