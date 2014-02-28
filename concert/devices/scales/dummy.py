"""Dummy scales."""
from concert.devices.scales import base
from concert.quantities import q


class Scales(base.Scales):

    """A dummy scale."""

    def __init__(self):
        super(Scales, self).__init__()
        self._weight = 147 * q.kg

    def _get_weight(self):
        return self._weight


class TarableScales(Scales):

    """A tarable dummy scale."""

    def _tare():
        pass
