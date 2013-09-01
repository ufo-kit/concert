import random
from concert.tests import assert_almost_equal
from concert.quantities import q
from concert.devices.base import LinearCalibration
from concert.devices.monochromators.dummy import\
    Monochromator as DummyMonochromator
from concert.devices.monochromators import base
from concert.devices.monochromators.base import Monochromator
from concert.tests.base import ConcertTest


class WavelengthMonochromator(Monochromator):

    """
    A monochromator which implements wavelength getter and setter. The
    conversion needs to be handled in the base class.
    """

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


class TestDummyMonochromator(ConcertTest):

    def setUp(self):
        super(TestDummyMonochromator, self).setUp()
        calibration = LinearCalibration(1 * q.eV, 0 * q.eV)
        self.mono = DummyMonochromator(calibration)
        self.wave_mono = WavelengthMonochromator()
        self.useless_mono = UselessMonochromator()
        self.energy = 25 * q.keV
        self.wavelength = 0.1 * q.nm

    def test_useless_mono_energy(self):
        def query_energy():
            self.useless_mono.energy

        def set_energy():
            self.useless_mono.energy = 25 * q.keV

        self.assertRaises(NotImplementedError, query_energy)
        self.assertRaises(NotImplementedError, set_energy)

    def test_useless_mono_wavelength(self):
        def query_wavelength():
            self.useless_mono.energy

        def set_wavelength():
            self.useless_mono.wavelength = 1e-10 * q.m

        self.assertRaises(NotImplementedError, query_wavelength)
        self.assertRaises(NotImplementedError, set_wavelength)

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
