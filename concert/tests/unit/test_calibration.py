from concert.quantities import q
from concert.devices.base import Calibration, LinearCalibration
from concert.tests.base import ConcertTest


class TestLinearCalibration(ConcertTest):
    STEPS_PER_UNIT = 5000 * q.count / q.mm
    OFFSET = 0 * q.mm

    def setUp(self):
        super(TestLinearCalibration, self).setUp()
        self.calibration = LinearCalibration(self.STEPS_PER_UNIT, self.OFFSET)

    def test_not_implemented(self):
        calibration = Calibration(q.count, q.mm)

        self.assertRaises(NotImplementedError, calibration.to_user,
                          1 * q.count)

        self.assertRaises(NotImplementedError, calibration.to_device, 2 * q.mm)

    def test_to_user(self):
        user = self.calibration.to_user(5000 * q.count)
        self.assertEqual(user, 1 * q.mm + self.OFFSET)

    def test_to_device(self):
        steps = self.calibration.to_device(1 * q.mm)
        self.assertEqual(steps, 5000 * q.count)

    def test_different_units(self):
        value = 1 * q.m
        steps = self.calibration.to_device(value)
        str_steps = "%.1f" % steps.magnitude
        expected = self.STEPS_PER_UNIT.to_base_units().magnitude
        self.assertEqual(str_steps, "%.1f" % expected)
