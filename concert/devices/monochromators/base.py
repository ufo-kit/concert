'''
Monochromator module. The device implementation needs to provide getters and
setters for wither wavelength or energy, it does not matter which one. The
conversion is handled in the base class.
'''
import quantities as q
import quantities.constants.quantum as cq
from concert.base import Parameter
from concert.devices.base import Device


def energy_to_wavelength(energy):
    """Convert *energy* [eV-like] to wavelength [m]."""
    res = cq.h * q.velocity.c / energy

    return res.rescale(q.m)


def wavelength_to_energy(wavelength):
    """Convert wavelength [m-like] to energy [eV]."""
    res = cq.h * q.velocity.c / wavelength

    return res.rescale(q.eV)


class Monochromator(Device):
    """Monochromator device which is used to filter the beam in order to
    get a very narrow energy bandwidth.

    .. py:attribute:: energy

        Monochromatic energy in electron volts.

    .. py:attribute:: wavelength

        Monochromatic wavelength in meters.
    """
    def __init__(self, calibration):
        params = [Parameter("energy", self.get_energy, self.set_energy,
                            q.eV, doc="Monochromatic energy"),
                  Parameter("wavelength", self.get_wavelength,
                            self.set_wavelength,
                            q.nanometer, doc="Monochromatic wavelength")]
        super(Monochromator, self).__init__(params)
        self._calibration = calibration

    def get_energy(self):
        # Check which method the subclass implements, use it and handle
        # conversions if necessary.
        if self.__class__.get_energy != Monochromator.get_energy:
            return self.get_energy()
        elif self.__class__.get_wavelength != Monochromator.get_wavelength:
            return wavelength_to_energy(self.get_wavelength())
        else:
            raise NotImplementedError

    def set_energy(self, energy):
        if self.__class__.set_energy != Monochromator.set_energy:
            self.set_energy(energy)
        elif self.__class__.set_wavelength != Monochromator.set_wavelength:
            self.set_wavelength(energy_to_wavelength(energy))
        else:
            raise NotImplementedError

    def get_wavelength(self):
        if self.__class__.get_energy != Monochromator.get_energy:
            return energy_to_wavelength(self.get_energy())
        elif self.__class__.get_wavelength != Monochromator.get_wavelength:
            return self.get_wavelength()
        else:
            raise NotImplementedError

    def set_wavelength(self, wavelength):
        if self.__class__.set_energy != Monochromator.set_energy:
            self.set_energy(wavelength_to_energy(wavelength))
        elif self.__class__.set_wavelength != Monochromator.set_wavelength:
            self.set_wavelength(wavelength)
        else:
            raise NotImplementedError
