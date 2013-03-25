import unittest
import logbook
import quantities as q
from concert.tests import slow
from concert.base import wait
from concert.devices.motors.base import LinearCalibration, Motor
from concert.devices.motors.dummy import DummyMotor, DummyContinuousMotor


class TestDummyMotor(unittest.TestCase):
    def setUp(self):
        calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        self.motor = DummyMotor(calibration)
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_set_position_blocking(self):
        position = 1 * q.mm
        self.motor.set_position(position, True)
        new_position = self.motor.get_position()
        self.assertEqual(position, new_position)

    def test_set_position_nonblocking(self):
        position = 1 * q.mm
        e = self.motor.set_position(position, False)
        wait([e])
        self.assertEqual(position, self.motor.get_position())

    def test_set_positions_nonblocking(self):
        motor1 = DummyMotor(LinearCalibration(1 / q.mm, 0 * q.mm))

        position = 1 * q.mm
        position1 = 3 * q.mm

        event_1 = self.motor.set_position(position, False)
        event_2 = motor1.set_position(position1, False)
        wait([event_1, event_2])
        self.assertEqual(position, self.motor.get_position())
        self.assertEqual(position1, motor1.get_position())

    def test_move(self):
        position = 1 * q.mm
        delta = 0.5 * q.mm
        self.motor.set_position(position, True)
        self.motor.move(delta, True)
        self.assertEqual(position + delta, self.motor.get_position())

    def test_log_output(self):
        self.motor.set_position(0 * q.mm, True)
        info = "DummyMotor: set position='0.0 mm' blocking='True'"
        self.assertTrue(self.handler.has_info(info))

        self.motor.set_position(2 * q.mm, False).wait()
        info = "DummyMotor: set position='2.0 mm' blocking='False'"
        self.assertTrue(self.handler.has_info(info))


class TestContinuousDummyMotor(unittest.TestCase):
    def setUp(self):
        position_calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        velocity_calibration = LinearCalibration(1 / (q.mm / q.s),
                                                 0 * (q.mm / q.s))

        self.motor = DummyContinuousMotor(position_calibration,
                                          velocity_calibration)

        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    @slow
    def test_set_velocity_blocking(self):
        velocity = 1 * q.mm / q.s
        self.motor.set_velocity(velocity, True)
        new_velocity = self.motor.get_velocity()
        self.assertEqual(velocity, new_velocity)


class TestMotorCalibration(unittest.TestCase):
    def setUp(self):
        self.steps_per_mm = 10. / q.mm
        calibration = LinearCalibration(self.steps_per_mm, 0 * q.mm)

        class MockMotor(Motor):
            def __init__(self):
                super(MockMotor, self).__init__(calibration)
                self._position = 0 * q.dimensionless

            def _stop_real(self):
                pass

            def _set_position(self, position):
                self._position = position

            def _get_position(self):
                return self._position

        self.motor = MockMotor()
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_set_position(self):
        position = 100 * q.mm
        steps = position * self.steps_per_mm

        self.motor.set_position(position, True)
        self.assertEqual(self.motor._position, steps)
        self.assertEqual(self.motor.get_position(), position)
