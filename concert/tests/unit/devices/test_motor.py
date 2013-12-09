from concert.quantities import q
from concert.base import HardLimitError
from concert.devices.motors.dummy import LinearMotor, ContinuousLinearMotor
from concert.devices.motors.dummy import RotationMotor, ContinuousRotationMotor
from concert.tests import TestCase, suppressed_logging


class TestMotor(TestCase):

    def setUp(self):
        super(TestMotor, self).setUp()
        self.motor = LinearMotor()

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
        self.motor.move(delta).join()
        self.assertEqual(position + delta, self.motor.position)


class TestContinuousLinearMotor(TestCase):

    def setUp(self):
        super(TestContinuousLinearMotor, self).setUp()
        self.motor = ContinuousLinearMotor()

    def test_set_position(self):
        position = 1 * q.mm
        self.motor.position = position
        self.assertEqual(position, self.motor.position)

    def test_set_velocity(self):
        velocity = 1 * q.mm / q.s
        self.motor.velocity = velocity
        self.assertEqual(velocity, self.motor.velocity)


class TestRotationMotor(TestCase):

    def setUp(self):
        super(TestRotationMotor, self).setUp()
        self.motor = RotationMotor()

    def test_set_position(self):
        position = 1 * q.deg
        self.motor.position = position
        self.assertEqual(position, self.motor.position)

    def test_move(self):
        position = 1 * q.deg
        delta = 0.5 * q.deg
        self.motor.position = position
        self.motor.move(delta).join()
        self.assertEqual(position + delta, self.motor.position)


class TestContinuousRotationMotor(TestCase):

    def setUp(self):
        super(TestContinuousRotationMotor, self).setUp()
        self.motor = ContinuousRotationMotor()

    def test_set_position(self):
        position = 1 * q.deg
        self.motor.position = position
        self.assertEqual(position, self.motor.position)

    def test_set_velocity(self):
        velocity = 1 * q.deg / q.s
        self.motor.velocity = velocity
        self.assertEqual(velocity, self.motor.velocity)
