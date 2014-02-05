"""Dummy scales."""
from concert.devices.scales import base


class Scales(base.Scales):
    def __init__(self):
        super(Scales, self).__init__()
        self._weight = 147

    def _get_weight(self):
        return self._weight


class TarableScales(Scales):
    def _tare():
        pass
