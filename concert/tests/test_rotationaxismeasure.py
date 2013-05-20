'''
Created on May 20, 2013

@author: farago
'''
import quantities as q
import numpy as np
import unittest
from concert.measures.rotationaxis import Ellipse
from concert.tests.util.rotationaxis import ImageSource
from concert.devices.motors.base import LinearCalibration
from concert.devices.motors.dummy import Motor
from concert.tests import slow


class TestRotationAxisMeasure(unittest.TestCase):
    def setUp(self):
        self.z_motor = Motor(LinearCalibration(1 / q.deg, 0 * q.deg),
                             hard_limits=(-1e5, 1e5))
        self.z_motor["position"].unit = q.deg
        self.x_motor = Motor(LinearCalibration(1 / q.deg, 0 * q.deg),
                             hard_limits=(-1e5, 1e5))
        self.x_motor["position"].unit = q.deg

        # The bigger the image size, the more images we need to determine
        # the center correctly.
        self.image_source = ImageSource(256, self.x_motor["position"],
                                        self.z_motor["position"], 50)
        self.measure = Ellipse()

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2.0/self.image_source.rotation_radius)*q.rad

    def make_images(self, x_angle, z_angle):
        self.x_motor.position = z_angle
        self.z_motor.position = x_angle
        self.measure.images = self.image_source.get_images()

    def align_check(self, x_angle, z_angle):
        self.make_images(x_angle, z_angle)
        phi, psi = self.measure()

        assert phi + x_angle < self.eps
        assert np.abs(psi) - np.abs(z_angle) < self.eps

    def center_check(self):
        assert np.abs(self.measure.center[1] - self.image_source.center[1]) < 1
        assert np.abs(self.measure.center[0] - self.image_source.center[0]) < 1

    @slow
    def test_center_no_rotation(self):
        self.make_images(0*q.deg, 0*q.deg)
        self.center_check()

    @slow
    def test_center_only_x(self):
        self.make_images(17*q.deg, 0*q.deg)
        self.center_check()

    @slow
    def test_center_only_z(self):
        self.make_images(0*q.deg, 11*q.deg)
        self.center_check()

    @slow
    def test_center_positive(self):
        self.make_images(17*q.deg, 11*q.deg)
        self.center_check()

    @slow
    def test_center_negative_positive(self):
        self.make_images(-17*q.deg, 11*q.deg)
        self.center_check()

    @slow
    def test_center_positive_negative(self):
        self.make_images(17*q.deg, -11*q.deg)
        self.center_check()

    @slow
    def test_center_negative(self):
        self.make_images(-17*q.deg, -11*q.deg)
        self.center_check()

    @slow
    def test_only_x(self):
        """Only misaligned laterally."""
        self.align_check(0*q.deg, 0*q.deg)

    @slow
    def test_only_z(self):
        """Only misaligned in the beam direction."""
        self.align_check(0*q.deg, 11*q.deg)

    @slow
    def test_huge_x(self):
        self.image_source.scale = (3, 0.25, 3)
        self.align_check(60*q.deg, 11*q.deg)

    @slow
    def test_huge_z(self):
        self.image_source.scale = (3, 0.25, 3)
        self.align_check(11*q.deg, 60*q.deg)

    @slow
    def test_positive(self):
        self.align_check(17*q.deg, 11*q.deg)

    @slow
    def test_z_ambiguity(self):
        self.align_check(17*q.deg, -11*q.deg)

    @slow
    def test_negative_positive(self):
        self.align_check(-17*q.deg, 11*q.deg)

    @slow
    def test_positive_negative(self):
        self.align_check(17*q.deg, -11*q.deg)

    @slow
    def test_negative(self):
        self.align_check(-17*q.deg, -11*q.deg)
