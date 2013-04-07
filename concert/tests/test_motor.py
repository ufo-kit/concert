import unittest
import logbook
import quantities as q
from concert.devices.motors.base import LinearCalibration, Motor
from concert.devices.motors.dummy import DummyMotor, DummyContinuousMotor


class TestDummyMotor(unittest.TestCase):
    def setUp(self):
        calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        self.motor = DummyMotor(calibration)
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_set_position(self):
        position = 1 * q.mm
        self.motor.position = position
        self.assertEqual(position, self.motor.position)

    def test_move(self):
        position = 1 * q.mm
        delta = 0.5 * q.mm
        self.motor.position = position
        self.motor.move(delta).wait()
        self.assertEqual(position + delta, self.motor.position)

    def test_log_output(self):
        self.motor.position = 0 * q.mm
        info = "DummyMotor: try position='0.0 mm'"
        self.assertTrue(self.handler.has_info(info))

        self.motor.position = 2 * q.mm
        info = "DummyMotor: try position='2.0 mm'"
        self.assertTrue(self.handler.has_info(info))


class TestContinuousDummyMotor(unittest.TestCase):
    def setUp(self):
        position_calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        velocity_calibration = LinearCalibration(1 / (q.mm / q.s),
                                                 0 * (q.mm / q.s))

        self.motor = DummyContinuousMotor(position_calibration,
                                          velocity_calibration)

        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_set_velocity(self):
        velocity = 1 * q.mm / q.s
        self.motor.velocity = velocity
        self.assertEqual(velocity, self.motor.velocity)


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
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_set_position(self):
        position = 100 * q.mm
        steps = position * self.steps_per_mm

        self.motor.position = position
        self.assertEqual(self.motor._position, steps)
        self.assertEqual(self.motor.position, position)
