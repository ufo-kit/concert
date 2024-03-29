'''
Monochromator module. The device implementation needs to provide getters and
setters for wither wavelength or energy, it does not matter which one. The
conversion is handled in the base class.
'''
from concert.quantities import q
from concert.base import Quantity, AccessorNotImplementedError, State, check
from concert.devices.base import Device

# pint supports constants and defines hbar like this, but I haven't found a way
# to get the constants ...
hbar = 6.62606957e-34 * q.J * q.s


def energy_to_wavelength(energy):
    """Convert *energy* [eV-like] to wavelength [m]."""
    res = hbar * q.c / energy
    res.ito(q.m)
    return res


def wavelength_to_energy(wavelength):
    """Convert wavelength [m-like] to energy [eV]."""
    res = hbar * q.c / wavelength
    res.ito(q.eV)
    return res


class Monochromator(Device):

    """Monochromator device which is used to filter the beam in order to
    get a very narrow energy bandwidth.

    .. py:attribute:: energy

        Monochromatic energy in electron volts.

    .. py:attribute:: wavelength

        Monochromatic wavelength in meters.
    """

    state = State(default="standby")
    energy = Quantity(q.eV, help="Energy")
    wavelength = Quantity(q.nanometer, help="Wavelength")

    async def _get_energy(self):
        try:
            return await self._get_energy_real()
        except AccessorNotImplementedError:
            return wavelength_to_energy(await self._get_wavelength_real())

    @check(source="standby", target="standby")
    async def _set_energy(self, energy):
        try:
            await self._set_energy_real(energy)
        except AccessorNotImplementedError:
            await self._set_wavelength_real(energy_to_wavelength(energy))

    async def _get_wavelength(self):
        try:
            return await self._get_wavelength_real()
        except AccessorNotImplementedError:
            return energy_to_wavelength(await self._get_energy_real())

    @check(source="standby", target="standby")
    async def _set_wavelength(self, wavelength):
        try:
            await self._set_wavelength_real(wavelength)
        except AccessorNotImplementedError:
            await self._set_energy_real(wavelength_to_energy(wavelength))

    async def _get_energy_real(self):
        raise AccessorNotImplementedError

    async def _set_energy_real(self, energy):
        raise AccessorNotImplementedError

    async def _get_wavelength_real(self):
        raise AccessorNotImplementedError

    async def _set_wavelength_real(self, wavelength):
        raise AccessorNotImplementedError
