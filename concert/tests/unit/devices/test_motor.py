from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor, ContinuousLinearMotor
from concert.devices.motors.dummy import RotationMotor, ContinuousRotationMotor
from concert.tests import TestCase, assert_almost_equal


class TestMotor(TestCase):

    def setUp(self):
        super(TestMotor, self).setUp()
        self.motor = LinearMotor()

    def test_set_position(self):
        position = 1 * q.mm
        self.motor.position = position
        self.assertEqual(position, self.motor.position)

    async def test_move(self):
        position = 1 * q.mm
        delta = 0.5 * q.mm
        await self.motor.set_position(position)
        await self.motor.move(delta)
        self.assertEqual(position + delta, await self.motor.get_position())


class TestContinuousLinearMotor(TestCase):

    def setUp(self):
        super(TestContinuousLinearMotor, self).setUp()
        self.motor = ContinuousLinearMotor()

    def test_set_position(self):
        position = 1 * q.mm
        self.motor.position = position
        self.assertEqual(position, self.motor.position)
        self.assertEqual(self.motor.state, 'standby')

    async def test_set_velocity(self):
        velocity = 1 * q.mm / q.s
        await self.motor.set_velocity(velocity)
        assert_almost_equal(velocity, await self.motor.get_velocity(), 0.1)
        self.assertEqual(await self.motor.get_state(), 'moving')
        await self.motor.stop()


class TestRotationMotor(TestCase):

    def setUp(self):
        super(TestRotationMotor, self).setUp()
        self.motor = RotationMotor()

    def test_set_position(self):
        position = 1 * q.deg
        self.motor.position = position
        self.assertEqual(position, self.motor.position)
        self.assertEqual(self.motor.state, 'standby')

    async def test_move(self):
        position = 1 * q.deg
        delta = 0.5 * q.deg
        await self.motor.set_position(position)
        await self.motor.move(delta)
        self.assertEqual(position + delta, await self.motor.get_position())
        self.assertEqual(await self.motor.get_state(), 'standby')


class TestContinuousRotationMotor(TestCase):

    def setUp(self):
        super(TestContinuousRotationMotor, self).setUp()
        self.motor = ContinuousRotationMotor()

    def test_set_position(self):
        position = 1 * q.deg
        self.motor.position = position
        self.assertEqual(position, self.motor.position)

    async def test_set_velocity(self):
        velocity = 1 * q.deg / q.s
        await self.motor.set_velocity(velocity)
        assert_almost_equal(velocity, await self.motor.get_velocity(), 0.1)
        self.assertEqual(await self.motor.get_state(), 'moving')
        await self.motor.stop()
