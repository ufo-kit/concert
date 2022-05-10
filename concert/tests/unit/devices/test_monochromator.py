import random
import numpy as np
from concert.tests import assert_almost_equal, TestCase, slow
from concert.quantities import q
from concert.devices.monochromators.dummy import\
    Monochromator as DummyMonochromator
from concert.devices.monochromators import base
from concert.devices.monochromators.base import Monochromator
from concert.devices.monochromators.dummy import DoubleMonochromator
from concert.devices.photodiodes.dummy import PhotoDiode as DummyPhotoDiode


class WavelengthMonochromator(Monochromator):

    """
    A monochromator which implements wavelength getter and setter. The
    conversion needs to be handled in the base class.
    """

    async def __ainit__(self):
        await super(WavelengthMonochromator, self).__ainit__()
        self._wavelength = random.random() * 1e-10 * q.m

    async def _get_wavelength_real(self):
        return self._wavelength

    async def _set_wavelength_real(self, wavelength):
        self._wavelength = wavelength


class PhotoDiode(DummyPhotoDiode):
    """
    Photo diode that returns an intensity distribution depending on the bragg_motor2 position.

    """
    async def __ainit__(self, bragg_motor2):
        self.bragg_motor = bragg_motor2
        self.function = None
        await super().__ainit__()

    async def _get_intensity(self):
        x = (await self.bragg_motor.get_position()).to(q.deg).magnitude
        return self.function(x) * q.V


class TestDummyMonochromator(TestCase):

    async def asyncSetUp(self):
        await super(TestDummyMonochromator, self).asyncSetUp()
        self.mono = await DummyMonochromator()
        self.wave_mono = await WavelengthMonochromator()
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


@slow
class TestDummyDoubleMonochromator(TestCase):
    async def asyncSetUp(self):
        self.mono = await DoubleMonochromator()
        self.diode = await PhotoDiode(self.mono._motor_2)

    def gaussian(self, x):
        """
        Gaussian centered around 0.2 with a sigma of 0.1.
        """
        mu = 0.2
        sigma = 0.1
        return np.exp(-(x - mu) ** 2 / sigma ** 2)

    def double_gaussian(self, x):
        """
        Double two gaussian functions centered around zero with a sigma of 0.2 each.
        """
        mu_1 = -0.2
        mu_2 = 0.2
        sigma = 0.2
        return np.exp(-(x - mu_1) ** 2 / sigma ** 2) + np.exp(-(x - mu_2) ** 2 / sigma ** 2)

    async def test_center(self):
        """
        This test configures the diode to return a gaussian profile with the center at 0.2 deg.
        Then it is checked if the monochromator._motor2 is moved to 0.2 deg after the scan and the
        select_maximum() function.
        """
        self.diode.function = self.gaussian
        await self.mono.scan_bragg_angle(diode=self.diode, tune_range=1 * q.deg, n_points=100)
        await self.mono.select_maximum()
        self.assertAlmostEqual(await self.mono._motor_2.get_position(), 0.2 * q.deg, 2)

    async def test_center_of_mass(self):
        """
        This test configures the diode to return a hat profile with the center at 0.0 deg.
        Then it is checked if the monochromator._motor2 is moved to 0.0 deg after the scan and the
        select_center_of_mass() function.
        """
        self.diode.function = self.double_gaussian
        await self.mono.scan_bragg_angle(diode=self.diode, tune_range=1 * q.deg, n_points=100)
        await self.mono.select_center_of_mass()
        self.assertAlmostEqual(await self.mono._motor_2.get_position(), 0.0 * q.deg, 2)
