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

    def test_positive_direction(self):
        # Test positive direction
        axis = Axis('x', self.motor, direction=1)
        axis.set_position(1 * q.mm).join()
        assert_almost_equal(self.motor.position, axis.get_position().result())

    def test_negative_direction(self):
        # Test positive direction
        axis = Axis('x', self.motor, direction=-1)
        axis.set_position(-1 * q.mm).join()
        assert_almost_equal(self.motor.position, - axis.get_position().result())


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
            self.positioner.set_position(position).join()

        # The remaining axes must work
        position = (0.0, 1.0, 2.0) * q.mm
        self.positioner.position = position
        assert_almost_equal(position[1:], self.positioner.position[1:])

        # Also nan must work
        position = (np.nan, 1.0, 2.0) * q.mm
        self.positioner.position = position
        assert_almost_equal(position, self.positioner.position)

        # Also 0 in the place of no axis must work
        self.positioner.position = (0.0, 1.0, 2.0) * q.mm
        assert_almost_equal(position, self.positioner.position)

    def test_orientation(self):
        orientation = (1.0, 2.0, 3.0) * q.rad
        self.positioner.orientation = orientation
        assert_almost_equal(orientation, self.positioner.orientation)

        # Degrees must be accepted
        self.positioner.orientation = (2.0, 3.0, 4.0) * q.deg

        # Test non-existent axis
        del self.positioner.rotators['x']
        with self.assertRaises(PositionerError):
            self.positioner.set_orientation(orientation).join()

        # Also nan must work
        orientation = (np.nan, 1.0, 2.0) * q.rad
        self.positioner.orientation = orientation
        assert_almost_equal(orientation, self.positioner.orientation)

        # Also 0 in the place of no axis must work
        self.positioner.orientation = (0.0, 1.0, 2.0) * q.rad
        assert_almost_equal(orientation, self.positioner.orientation)

    def test_move(self):
        position = (1.0, 2.0, 3.0) * q.mm
        self.positioner.move(position).join()
        assert_almost_equal(position, self.positioner.position)

    def test_rotate(self):
        orientation = (1.0, 2.0, 3.0) * q.rad
        self.positioner.rotate(orientation).join()
        assert_almost_equal(orientation, self.positioner.orientation)

    def test_right(self):
        self.positioner.right(1 * q.mm).join()
        assert_almost_equal((1.0, 0.0, 0.0) * q.mm, self.positioner.position)

    def test_left(self):
        self.positioner.left(1 * q.mm).join()
        assert_almost_equal((-1.0, 0.0, 0.0) * q.mm, self.positioner.position)

    def test_up(self):
        self.positioner.up(1 * q.mm).join()
        assert_almost_equal((0.0, 1.0, 0.0) * q.mm, self.positioner.position)

    def test_down(self):
        self.positioner.down(1 * q.mm).join()
        assert_almost_equal((0.0, -1.0, 0.0) * q.mm, self.positioner.position)

    def test_forward(self):
        self.positioner.forward(1 * q.mm).join()
        assert_almost_equal((0.0, 0.0, 1.0) * q.mm, self.positioner.position)

    def test_back(self):
        self.positioner.back(1 * q.mm).join()
        assert_almost_equal((0.0, 0.0, -1.0) * q.mm, self.positioner.position)


class TestImagingPositioner(TestCase):

    def setUp(self):
        super(TestImagingPositioner, self).setUp()
        self.positioner = ImagingPositioner()
        self.positioner.position = ORIGIN
        self.positioner.orientation = ROT_ORIGIN
        self._pixel_width_position = self.positioner.detector.pixel_width.to(q.m).magnitude
        self._pixel_height_position = self.positioner.detector.pixel_height.to(q.m).magnitude

    def test_move(self):
        pixel_width = self.positioner.detector.pixel_width.to(q.m).magnitude
        pixel_height = self.positioner.detector.pixel_height.to(q.m).magnitude

        self.positioner.move((100.0, 200.0, 0.0) * q.pixel).join()
        position = (100.0 * pixel_width, 200.0 * pixel_height, 0.0) * q.m
        assert_almost_equal(position, self.positioner.position)

        # Cannot move in z-direction by pixel size
        with self.assertRaises(PositionerError):
            self.positioner.move((1.0, 2.0, 3.0) * q.pixel).join()

    def test_right(self):
        self.positioner.right(1 * q.px).join()
        assert_almost_equal((self._pixel_width_position, 0.0, 0.0) * q.m, self.positioner.position)

    def test_left(self):
        self.positioner.left(1 * q.px).join()
        assert_almost_equal((- self._pixel_width_position, 0.0, 0.0) * q.m, self.positioner.position)

    def test_up(self):
        self.positioner.up(1 * q.px).join()
        assert_almost_equal((0.0, self._pixel_height_position, 0.0) * q.m, self.positioner.position)

    def test_down(self):
        self.positioner.down(1 * q.px).join()
        assert_almost_equal((0.0, - self._pixel_height_position, 0.0) * q.m, self.positioner.position)

    def test_forward(self):
        with self.assertRaises(PositionerError):
            self.positioner.forward(1 * q.px).join()

    def test_back(self):
        with self.assertRaises(PositionerError):
            self.positioner.back(1 * q.px).join()
