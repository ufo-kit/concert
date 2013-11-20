from concert.quantities import q
from concert.base import HardLimitError
from concert.devices.motors.base import Motor
from concert.devices.motors.dummy import Motor as DummyMotor,\
    ContinuousMotor as DummyContinuousMotor
from concert.tests import TestCase, suppressed_logging


class TestDummyMotor(TestCase):

    def setUp(self):
        super(TestDummyMotor, self).setUp()
        self.motor = DummyMotor()

    def test_set_position(self):
        position = 1 * q.mm
        self.motor.position = position
        self.assertEqual(position, self.motor.position)

    def test_hard_limit(self):
        with self.assertRaises(HardLimitError):
            self.motor.set_position(1e6 * q.m).result()

    def test_move(self):
        position = 1 * q.mm
        delta = 0.5 * q.mm
        self.motor.position = position
        self.motor.move(delta).wait()
        self.assertEqual(position + delta, self.motor.position)


class TestContinuousDummyMotor(TestCase):

    def setUp(self):
        super(TestContinuousDummyMotor, self).setUp()
        self.motor = DummyContinuousMotor()

    def test_set_velocity(self):
        velocity = 1 * q.deg / q.s
        self.motor.velocity = velocity
        self.assertEqual(velocity, self.motor.velocity)

    def test_hard_limit(self):
        with self.assertRaises(HardLimitError):
            self.motor.set_velocity(1e6 * q.deg / q.s).result()
