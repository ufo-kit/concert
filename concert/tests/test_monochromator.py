'''
Created on Apr 11, 2013

@author: farago
'''
import unittest
import logbook
from testfixtures import ShouldRaise
from concert.devices.motors.base import LinearCalibration
import quantities as q
from concert.devices.monochromators.dummy import\
    Monochromator as DummyMonochromator
from concert.devices.monochromators import base
from concert.devices.monochromators.base import Monochromator


class WavelengthMonochromator(Monochromator):
    """A monochromator which implements wavelength getter and setter. The
    conversion needs to be handled in the base class."""
    def __init__(self):
        super(WavelengthMonochromator, self).__init__(self)
        self._wavelength = None

    def _get_wavelength(self):
        return self._wavelength

    def _set_wavelength(self, wavelength):
        self._wavelength = wavelength


class UselessMonochromator(Monochromator):
    """A monochromator wihch does not implement anything."""
    def __init__(self):
        super(UselessMonochromator, self).__init__(self)


class TestDummyMonochromator(unittest.TestCase):
    def setUp(self):
        calibration = LinearCalibration(1*q.eV, 0*q.eV)
        self.mono = DummyMonochromator(calibration)
        self.wave_mono = WavelengthMonochromator()
        self.useless_mono = UselessMonochromator()
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_energy(self):
        energy = 25*q.keV

        # Dummy monochromator.
        self.mono.energy = energy
        self.assertAlmostEqual(self.mono.energy, energy)
        self.assertAlmostEqual(self.mono.wavelength,
                               base.energy_to_wavelength(self.mono.energy))

        # Wavelength-based monochromator
        self.wave_mono.energy = energy
        self.assertAlmostEqual(self.wave_mono.energy, energy)
        self.assertAlmostEqual(self.wave_mono.wavelength, base.
                               energy_to_wavelength(self.wave_mono.energy))

        # Useless monochromator.
        with ShouldRaise(NotImplementedError):
            self.useless_mono.energy = energy

    def test_wavelength(self):
        lam = 0.1*q.nm

        # Dummy monochromator.
        self.mono.wavelength = lam
        self.assertAlmostEqual(self.mono.wavelength, lam)
        self.assertAlmostEqual(base.wavelength_to_energy(lam),
                               self.mono.energy)

        # Wavelength-based monochromator.
        self.wave_mono.wavelength = lam
        self.assertAlmostEqual(self.wave_mono.wavelength, lam)
        self.assertAlmostEqual(base.wavelength_to_energy(lam),
                               self.wave_mono.energy)

        # Useless monochromator.
        with ShouldRaise(NotImplementedError):
            self.useless_mono.wavelength = lam
