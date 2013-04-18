'''
Created on Apr 11, 2013

@author: farago
'''
import unittest
from concert.devices.motors.base import LinearCalibration
import quantities as q
from concert.devices.monochromators.dummy import\
    Monochromator as DummyMonochromator
from concert.devices.monochromators import base


class TestDummyMonochromator(unittest.TestCase):
    def setUp(self):
        calibration = LinearCalibration(1*q.eV, 0*q.eV)
        self.mono = DummyMonochromator(calibration)

    def test_energy(self):
        energy = 25*q.keV
        self.mono.energy = energy
        self.assertAlmostEqual(self.mono.energy, energy)
        self.assertAlmostEqual(self.mono.wavelength,
                               base.energy_to_wavelength(self.mono.energy))

    def test_wavelength(self):
        lam = 0.1*q.nm

        self.mono.wavelength = lam
        self.assertAlmostEqual(self.mono.wavelength, lam)
        self.assertAlmostEqual(base.wavelength_to_energy(lam),
                               self.mono.energy)
