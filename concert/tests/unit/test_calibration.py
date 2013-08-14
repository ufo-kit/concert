import unittest
from concert.quantities import q
from concert.devices.motors.base import LinearCalibration


class TestLinearCalibration(unittest.TestCase):
    STEPS_PER_UNIT = 5000
    OFFSET = 0 * q.mm

    def setUp(self):
        self.calibration = LinearCalibration(self.STEPS_PER_UNIT * q.count /
                                             q.mm,
                                             self.OFFSET)

    def test_to_user(self):
        user = self.calibration.to_user(self.STEPS_PER_UNIT)
        self.assertEqual(user, 1 * q.mm + self.OFFSET)

    def test_to_steps(self):
        steps = self.calibration.to_steps(1 * q.mm)
        self.assertEqual(steps, self.STEPS_PER_UNIT)

    def test_different_units(self):
        value = 1 * q.m
        steps = self.calibration.to_steps(value)
        str_steps = "%.1f" % steps

        self.assertEqual(str_steps, "%.1f" % (self.STEPS_PER_UNIT * 1e3))
