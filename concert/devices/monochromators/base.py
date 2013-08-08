'''
Monochromator module. The device implementation needs to provide getters and
setters for wither wavelength or energy, it does not matter which one. The
conversion is handled in the base class.
'''
from concert.quantities import q
from concert.base import Parameter
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

    def __init__(self, calibration):
        params = [Parameter("energy", self._get_energy, self._set_energy,
                            q.eV, doc="Monochromatic energy"),
                  Parameter("wavelength", self._get_wavelength,
                            self._set_wavelength,
                            q.nanometer, doc="Monochromatic wavelength")]
        super(Monochromator, self).__init__(params)
        self._calibration = calibration

    def _get_energy(self):
        # Check which method the subclass implements, use it and handle
        # conversions if necessary.
        if self.__class__._get_energy != Monochromator._get_energy:
            return self._get_energy()
        elif self.__class__._get_wavelength != Monochromator._get_wavelength:
            return wavelength_to_energy(self._get_wavelength())
        else:
            raise NotImplementedError

    def _set_energy(self, energy):
        if self.__class__._set_energy != Monochromator._set_energy:
            self._set_energy(energy)
        elif self.__class__._set_wavelength != Monochromator._set_wavelength:
            self._set_wavelength(energy_to_wavelength(energy))
        else:
            raise NotImplementedError

    def _get_wavelength(self):
        if self.__class__._get_energy != Monochromator._get_energy:
            return energy_to_wavelength(self._get_energy())
        elif self.__class__._get_wavelength != Monochromator._get_wavelength:
            return self._get_wavelength()
        else:
            raise NotImplementedError

    def _set_wavelength(self, wavelength):
        if self.__class__._set_energy != Monochromator._set_energy:
            self._set_energy(wavelength_to_energy(wavelength))
        elif self.__class__._set_wavelength != Monochromator._set_wavelength:
            self._set_wavelength(wavelength)
        else:
            raise NotImplementedError
