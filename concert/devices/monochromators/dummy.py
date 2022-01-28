"""Monochromator Dummy"""
from concert.quantities import q
from concert.devices.monochromators import base
from concert.devices.monochromators import doublemonochromator
from concert.devices.motors.dummy import RotationMotor


class Monochromator(base.Monochromator):

    """Monochromator class implementation."""

    def __init__(self):
        super(Monochromator, self).__init__()
        self._energy = 100 * q.keV

    async def _get_energy_real(self):
        return self._energy

    async def _set_energy_real(self, energy):
        self._energy = energy


class DoubleMonochromator(doublemonochromator.Monochromator):
    """
    Double monochromator implementation
    """

    def __init__(self):
        dummy_motor = RotationMotor()
        dummy_motor.position = 0 * q.deg
        super().__init__(dummy_motor)
        self._energy = 100 * q.keV

    async def _get_energy_real(self):
        return self._energy

    async def _set_energy_real(self, energy):
        self._energy = energy
