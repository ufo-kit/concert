import numpy as np
from concert.tests import TestCase, assert_almost_equal
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor
from concert.devices.positioners.base import PositionerError, Axis
from concert.devices.positioners.dummy import Positioner, ImagingPositioner


ORIGIN = (0.0, 0.0, 0.0) * q.mm
ROT_ORIGIN = (0.0, 0.0, 0.0) * q.rad


class TestAxis(TestCase):

    async def asyncSetUp(self):
        await super(TestAxis, self).asyncSetUp()
        self.motor = await LinearMotor()
        await self.motor.set_position(0 * q.mm)

    async def test_positive_direction(self):
        # Test positive direction
        axis = await Axis('x', self.motor, direction=1)
        await axis.set_position(1 * q.mm)
        assert_almost_equal(await self.motor.get_position(), await axis.get_position())

    async def test_negative_direction(self):
        # Test positive direction
        axis = await Axis('x', self.motor, direction=-1)
        await axis.set_position(-1 * q.mm)
        assert_almost_equal(await self.motor.get_position(), - (await axis.get_position()))


class TestPositioners(TestCase):

    async def asyncSetUp(self):
        await super(TestPositioners, self).asyncSetUp()
        self.positioner = await Positioner()
        await self.positioner.set_position(ORIGIN)
        await self.positioner.set_orientation(ROT_ORIGIN)

    async def test_position(self):
        position = (1.0, 2.0, 3.0) * q.um
        await self.positioner.set_position(position)
        assert_almost_equal(position, await self.positioner.get_position())

        # Test non-existent axis
        del self.positioner.translators['x']
        with self.assertRaises(PositionerError):
            await self.positioner.set_position(position)

        # The remaining axes must work
        position = (0.0, 1.0, 2.0) * q.mm
        await self.positioner.set_position(position)
        assert_almost_equal(position[1:], (await self.positioner.get_position())[1:])

        # Also nan must work
        position = (np.nan, 1.0, 2.0) * q.mm
        await self.positioner.set_position(position)
        assert_almost_equal(position[1:], (await self.positioner.get_position())[1:])

        # Also 0 in the place of no axis must work
        await self.positioner.set_position((0.0, 1.0, 2.0) * q.mm)
        assert_almost_equal(position[1:], (await self.positioner.get_position())[1:])

    async def test_orientation(self):
        orientation = (1.0, 2.0, 3.0) * q.rad
        await self.positioner.set_orientation(orientation)
        assert_almost_equal(orientation, await self.positioner.get_orientation())

        # Degrees must be accepted
        await self.positioner.set_orientation((2.0, 3.0, 4.0) * q.deg)

        # Test non-existent axis
        del self.positioner.rotators['x']
        with self.assertRaises(PositionerError):
            await self.positioner.set_orientation(orientation)

        # Also nan must work
        orientation = (np.nan, 1.0, 2.0) * q.rad
        await self.positioner.set_orientation(orientation)
        # assert_almost_equal(orientation[1:], self.positioner.orientation[1:])

        # Also 0 in the place of no axis must work
        await self.positioner.set_orientation((0.0, 1.0, 2.0) * q.rad)
        # assert_almost_equal(orientation[1:], self.positioner.orientation[1:])

    async def test_move(self):
        position = (1.0, 2.0, 3.0) * q.mm
        await self.positioner.move(position)
        assert_almost_equal(position, await self.positioner.get_position())

    async def test_rotate(self):
        orientation = (1.0, 2.0, 3.0) * q.rad
        await self.positioner.rotate(orientation)
        assert_almost_equal(orientation, await self.positioner.get_orientation())

    async def test_right(self):
        await self.positioner.right(1 * q.mm)
        assert_almost_equal((1.0, 0.0, 0.0) * q.mm, await self.positioner.get_position())

    async def test_left(self):
        await self.positioner.left(1 * q.mm)
        assert_almost_equal((-1.0, 0.0, 0.0) * q.mm, await self.positioner.get_position())

    async def test_up(self):
        await self.positioner.up(1 * q.mm)
        assert_almost_equal((0.0, 1.0, 0.0) * q.mm, await self.positioner.get_position())

    async def test_down(self):
        await self.positioner.down(1 * q.mm)
        assert_almost_equal((0.0, -1.0, 0.0) * q.mm, await self.positioner.get_position())

    async def test_forward(self):
        await self.positioner.forward(1 * q.mm)
        assert_almost_equal((0.0, 0.0, 1.0) * q.mm, await self.positioner.get_position())

    async def test_back(self):
        await self.positioner.back(1 * q.mm)
        assert_almost_equal((0.0, 0.0, -1.0) * q.mm, await self.positioner.get_position())


class TestImagingPositioner(TestCase):

    async def asyncSetUp(self):
        await super(TestImagingPositioner, self).asyncSetUp()
        self.positioner = await ImagingPositioner()
        await self.positioner.set_position(ORIGIN)
        await self.positioner.set_orientation(ROT_ORIGIN)
        self._pixel_width_position = (
            await self.positioner.detector.get_pixel_width()
        ).to(q.m).magnitude
        self._pixel_height_position = (
            await self.positioner.detector.get_pixel_height()
        ).to(q.m).magnitude

    async def test_move(self):
        pixel_width = (await self.positioner.detector.get_pixel_width()).to(q.m).magnitude
        pixel_height = (await self.positioner.detector.get_pixel_height()).to(q.m).magnitude

        await self.positioner.move((100.0, 200.0, 0.0) * q.pixel)
        position = (100.0 * pixel_width, 200.0 * pixel_height, 0.0) * q.m
        assert_almost_equal(position, await self.positioner.get_position())

        # Cannot move in z-direction by pixel size
        with self.assertRaises(PositionerError):
            await self.positioner.move((1.0, 2.0, 3.0) * q.pixel)

    async def test_right(self):
        await self.positioner.right(1 * q.px)
        assert_almost_equal((self._pixel_width_position, 0.0, 0.0) * q.m,
                            await self.positioner.get_position())

    async def test_left(self):
        await self.positioner.left(1 * q.px)
        assert_almost_equal((- self._pixel_width_position, 0.0, 0.0) * q.m,
                            await self.positioner.get_position())

    async def test_up(self):
        await self.positioner.up(1 * q.px)
        assert_almost_equal((0.0, self._pixel_height_position, 0.0) * q.m,
                            await self.positioner.get_position())

    async def test_down(self):
        await self.positioner.down(1 * q.px)
        assert_almost_equal((0.0, - self._pixel_height_position, 0.0) * q.m,
                            await self.positioner.get_position())

    async def test_forward(self):
        with self.assertRaises(PositionerError):
            await self.positioner.forward(1 * q.px)

    async def test_back(self):
        with self.assertRaises(PositionerError):
            await self.positioner.back(1 * q.px)
