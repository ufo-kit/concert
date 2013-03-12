import unittest
import quantities as q
from concert.devices.axes.base import LinearCalibration


class TestLinearCalibration(unittest.TestCase):
    STEPS_PER_UNIT = 5000 * q.dimensionless
    OFFSET = 0 * q.mm 

    def setUp(self):
        self.calibration = LinearCalibration(self.STEPS_PER_UNIT / q.mm,
                                             self.OFFSET)

    def test_to_user(self):
        user = self.calibration.to_user(self.STEPS_PER_UNIT)
        self.assertEqual(user, 1 * q.mm + self.OFFSET)

    def test_to_steps(self):
        steps = self.calibration.to_steps(1 * q.mm)
        self.assertEqual(steps, self.STEPS_PER_UNIT)
