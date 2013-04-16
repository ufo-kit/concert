'''
Created on Apr 11, 2013

@author: farago
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
        params = [Parameter("energy", self._get_energy, self._set_energy,
                            q.eV, doc="Monochromatic energy"),
                  Parameter("wavelength", self._get_wavelength,
                            self._set_wavelength,
                            q.m, doc="Monochromatic wavelength")]
        super(Monochromator, self).__init__(params)
        self._calibration = calibration

    def _get_energy(self):
        raise NotImplementedError

    def _set_energy(self, energy):
        raise NotImplementedError

    def _get_wavelength(self):
        return energy_to_wavelength(self._get_energy())

    def _set_wavelength(self, wavelength):
        self._set_energy(wavelength_to_energy(wavelength))
