"""Monochromator Dummy"""
from concert.quantities import q
from concert.devices.monochromators import base


class Monochromator(base.Monochromator):

    """Monochromator class implementation."""

    def __init__(self):
        super(Monochromator, self).__init__()
        self._energy = 100 * q.keV

    def _get_energy_real(self):
        return self._energy

    def _set_energy_real(self, energy):
        self._energy = energy
