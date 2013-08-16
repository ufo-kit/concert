import unittest
import logbook
import random
from testfixtures import ShouldRaise
from concert.tests import assert_almost_equal
from concert.quantities import q
from concert.devices.base import LinearCalibration
from concert.devices.monochromators.dummy import\
    Monochromator as DummyMonochromator
from concert.devices.monochromators import base
from concert.devices.monochromators.base import Monochromator


class WavelengthMonochromator(Monochromator):

    """A monochromator which implements wavelength getter and setter. The
    conversion needs to be handled in the base class."""

    def __init__(self):
        super(WavelengthMonochromator, self).__init__(self)
        self._wavelength = random.random() * 1e-10 * q.m

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
        calibration = LinearCalibration(1 * q.eV, 0 * q.eV)
        self.mono = DummyMonochromator(calibration)
        self.wave_mono = WavelengthMonochromator()
        self.useless_mono = UselessMonochromator()
        self.energy = 25 * q.keV
        self.wavelength = 0.1 * q.nm
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_useless_mono_energy(self):
        with ShouldRaise(NotImplementedError):
            self.useless_mono.energy
        with ShouldRaise(NotImplementedError):
            self.useless_mono.energy = 25 * q.keV

    def test_useless_mono_wavelength(self):
        with ShouldRaise(NotImplementedError):
            self.useless_mono.wavelength
        with ShouldRaise(NotImplementedError):
            self.useless_mono.wavelength = 1e-10 * q.m

    def test_energy_mono_energy(self):
        self.mono.energy = self.energy
        assert_almost_equal(self.mono.energy, self.energy)
        assert_almost_equal(self.mono.wavelength,
                            base.energy_to_wavelength(self.mono.energy))

    def test_energy_mono_wavelength(self):
        self.mono.wavelength = self.wavelength
        assert_almost_equal(self.mono.wavelength, self.wavelength)
        assert_almost_equal(base.wavelength_to_energy(self.wavelength),
                            self.mono.energy)

    def test_wavelength_mono_energy(self):
        self.wave_mono.energy = self.energy
        assert_almost_equal(self.wave_mono.energy, self.energy)
        assert_almost_equal(self.wave_mono.wavelength,
                            base.energy_to_wavelength(self.wave_mono.energy))

    def test_wavelength_mono_wavelength(self):
        # Wavelength-based monochromator.
        self.wave_mono.wavelength = self.wavelength
        assert_almost_equal(self.wave_mono.wavelength, self.wavelength)
        assert_almost_equal(base.wavelength_to_energy(self.wavelength),
                            self.wave_mono.energy)
