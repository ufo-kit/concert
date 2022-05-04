"""Monochromator Dummy"""
from concert.quantities import q
from concert.devices.monochromators import base
from concert.devices.monochromators import doublemonochromator
from concert.devices.motors.dummy import RotationMotor


class Monochromator(base.Monochromator):

    """Monochromator class implementation."""

    async def __ainit__(self):
        await super(Monochromator, self).__ainit__()
        self._energy = 100 * q.keV

    async def _get_energy_real(self):
        return self._energy

    async def _set_energy_real(self, energy):
        self._energy = energy


class DoubleMonochromator(doublemonochromator.Monochromator):
    """
    Double monochromator implementation
    """

    async def __ainit__(self):
        dummy_motor = await RotationMotor()
        await dummy_motor.set_position(0 * q.deg)
        await super().__ainit__(dummy_motor)
        self._energy = 100 * q.keV

    async def _get_energy_real(self):
        return self._energy

    async def _set_energy_real(self, energy):
        self._energy = energy
