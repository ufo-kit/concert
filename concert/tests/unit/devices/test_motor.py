from concert.quantities import q
from concert.devices.base import LinearCalibration
from concert.devices.motors.base import Motor
from concert.devices.motors.dummy import Motor as DummyMotor,\
    ContinuousMotor as DummyContinuousMotor
from concert.tests.base import ConcertTest, suppressed_logging


@suppressed_logging
def test_default_motor_has_default_calibration():
    class MockMotor(Motor):

        def __init__(self):
            self._position = 0 * q.count
            super(MockMotor, self).__init__()

        def _set_position(self, position):
            self._position = position

        def _get_position(self):
            return self._position

    motor = MockMotor()
    motor.move(-1 * q.mm).wait()
    assert motor.position == -1 * q.mm
    motor.position = 2.3 * q.mm
    assert motor.position == 2.3 * q.mm


@suppressed_logging
def test_different_calibration_unit():
    calibration = LinearCalibration(q.count / q.deg, 0 * q.deg)

    motor = DummyMotor(calibration)
    motor.position = 0 * q.deg
    motor.move(1 * q.deg).wait()
    assert motor.position == 1 * q.deg


class TestDummyMotor(ConcertTest):

    def setUp(self):
        super(TestDummyMotor, self).setUp()
        self.motor = DummyMotor()

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


class TestContinuousDummyMotor(ConcertTest):

    def setUp(self):
        super(TestContinuousDummyMotor, self).setUp()
        position_calibration = LinearCalibration(q.count / q.mm, 0 * q.mm)
        velocity_calibration = LinearCalibration(q.count / (q.mm / q.s),
                                                 0 * (q.mm / q.s))

        self.motor = DummyContinuousMotor(position_calibration,
                                          velocity_calibration)

    def test_set_velocity(self):
        velocity = 1 * q.mm / q.s
        self.motor.velocity = velocity
        self.assertEqual(velocity, self.motor.velocity)


class TestMotorCalibration(ConcertTest):

    def setUp(self):
        super(TestMotorCalibration, self).setUp()
        self.steps_per_mm = 10. * q.count / q.mm
        calibration = LinearCalibration(self.steps_per_mm, 0 * q.mm)

        class MockMotor(Motor):

            def __init__(self):
                super(MockMotor, self).__init__(calibration)
                self._position = 0 * q.count

            def _stop_real(self):
                pass

            def _set_position(self, position):
                self._position = position

            def _get_position(self):
                return self._position

        self.motor = MockMotor()

    def test_set_position(self):
        position = 100 * q.mm
        steps = position * self.steps_per_mm

        self.motor.position = position
        self.assertEqual(self.motor._position, steps)
        self.assertEqual(self.motor.position, position)
