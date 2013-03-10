import unittest
import quantities as q

from control.devices.motion.axes.calibration import LinearCalibration
from control.devices.motion.axes.dummyaxis import DummyAxis, DummyContinuousAxis
from control.devices.motion.axes.axis import AxisState, ContinuousAxisState


class TestDummyAxis(unittest.TestCase):
    def setUp(self):
        calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        self.axis = DummyAxis(None, calibration)

    def test_set_position_blocking(self):
        position = 1 * q.mm
        self.axis.set_position(position, True)
        new_position = self.axis.get_position()
        self.assertEqual(position, new_position)

    def test_set_position_nonblocking(self):
        position = 1 * q.mm

        def check_position(axis):
            new_position = axis.get_position()
            self.assertEqual(position, new_position)

        self.axis.set_position(position, False)
        self.axis.wait_for(AxisState.STANDBY, check_position)


class TestContinuousDummyAxis(unittest.TestCase):
    def setUp(self):
        position_calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        velocity_calibration = LinearCalibration(1 / (q.mm / q.s),
                                                 0 * (q.mm / q.s))

        self.axis = DummyContinuousAxis(None,
                                        position_calibration,
                                        velocity_calibration,
                                        None, None)

    def test_set_velocity_blocking(self):
        velocity = 1 * q.mm / q.s
        self.axis.set_velocity(velocity, True)
        new_velocity = self.axis.get_velocity()
        self.assertEqual(velocity, new_velocity)
