import numpy as np
from concert.tests import TestCase, assert_almost_equal
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor
from concert.devices.positioners.base import PositionerError, Axis
from concert.devices.positioners.dummy import Positioner, ImagingPositioner


ORIGIN = (0.0, 0.0, 0.0) * q.mm
ROT_ORIGIN = (0.0, 0.0, 0.0) * q.rad


class TestAxis(TestCase):

    def setUp(self):
        super(TestAxis, self).setUp()
        self.motor = LinearMotor()
        self.motor.position = 0 * q.mm

    async def test_positive_direction(self):
        # Test positive direction
        axis = Axis('x', self.motor, direction=1)
        await axis.set_position(1 * q.mm)
        assert_almost_equal(await self.motor.get_position(), await axis.get_position())

    async def test_negative_direction(self):
        # Test positive direction
        axis = Axis('x', self.motor, direction=-1)
        await axis.set_position(-1 * q.mm)
        assert_almost_equal(await self.motor.get_position(), - (await axis.get_position()))


class TestPositioners(TestCase):

    def setUp(self):
        super(TestPositioners, self).setUp()
        self.positioner = Positioner()
        self.positioner.position = ORIGIN
        self.positioner.orientation = ROT_ORIGIN

    def test_position(self):
        position = (1.0, 2.0, 3.0) * q.um
        self.positioner.position = position
        assert_almost_equal(position, self.positioner.position)

        # Test non-existent axis
        del self.positioner.translators['x']
        with self.assertRaises(PositionerError):
            self.positioner.position = position

        # The remaining axes must work
        position = (0.0, 1.0, 2.0) * q.mm
        self.positioner.position = position
        assert_almost_equal(position[1:], self.positioner.position[1:])

        # Also nan must work
        position = (np.nan, 1.0, 2.0) * q.mm
        self.positioner.position = position
        assert_almost_equal(position[1:], self.positioner.position[1:])

        # Also 0 in the place of no axis must work
        self.positioner.position = (0.0, 1.0, 2.0) * q.mm
        assert_almost_equal(position[1:], self.positioner.position[1:])

    def test_orientation(self):
        orientation = (1.0, 2.0, 3.0) * q.rad
        self.positioner.orientation = orientation
        assert_almost_equal(orientation, self.positioner.orientation)

        # Degrees must be accepted
        self.positioner.orientation = (2.0, 3.0, 4.0) * q.deg

        # Test non-existent axis
        del self.positioner.rotators['x']
        with self.assertRaises(PositionerError):
            self.positioner.orientation = orientation

        # Also nan must work
        orientation = (np.nan, 1.0, 2.0) * q.rad
        self.positioner.orientation = orientation
        # assert_almost_equal(orientation[1:], self.positioner.orientation[1:])

        # Also 0 in the place of no axis must work
        self.positioner.orientation = (0.0, 1.0, 2.0) * q.rad
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

    def setUp(self):
        super(TestImagingPositioner, self).setUp()
        self.positioner = ImagingPositioner()
        self.positioner.position = ORIGIN
        self.positioner.orientation = ROT_ORIGIN
        self._pixel_width_position = self.positioner.detector.pixel_width.to(q.m).magnitude
        self._pixel_height_position = self.positioner.detector.pixel_height.to(q.m).magnitude

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
