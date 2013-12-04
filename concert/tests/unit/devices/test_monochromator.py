import random
from concert.tests import assert_almost_equal, TestCase
from concert.quantities import q
from concert.devices.monochromators.dummy import\
    Monochromator as DummyMonochromator
from concert.devices.monochromators import base
from concert.devices.monochromators.base import Monochromator


class WavelengthMonochromator(Monochromator):

    """
    A monochromator which implements wavelength getter and setter. The
    conversion needs to be handled in the base class.
    """

    def __init__(self):
        super(WavelengthMonochromator, self).__init__()
        self._wavelength = random.random() * 1e-10 * q.m

    def _get_wavelength_real(self):
        return self._wavelength

    def _set_wavelength_real(self, wavelength):
        self._wavelength = wavelength


class TestDummyMonochromator(TestCase):

    def setUp(self):
        super(TestDummyMonochromator, self).setUp()
        self.mono = DummyMonochromator()
        self.wave_mono = WavelengthMonochromator()
        self.energy = 25 * q.keV
        self.wavelength = 0.1 * q.nm

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
