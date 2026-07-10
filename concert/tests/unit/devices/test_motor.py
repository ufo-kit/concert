from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

from concert.quantities import q
from concert.base import Parameterizable
from concert.devices.base import Device
from concert.devices.motors import base
from concert.devices.motors import dummy
from concert.devices.motors.dummy import LinearMotor, ContinuousLinearMotor
from concert.devices.motors.dummy import RotationMotor, ContinuousRotationMotor
from concert.tests import TestCase, assert_almost_equal


class OffsetLinearMotor(LinearMotor):

    async def _get_position(self):
        return await super()._get_position() + 1 * q.mm


class OffsetContinuousLinearMotor(ContinuousLinearMotor):

    async def _get_position(self):
        return await super()._get_position() + 1 * q.mm


class TestMotor(TestCase):

    async def asyncSetUp(self):
        await super(TestMotor, self).asyncSetUp()
        self.motor = await LinearMotor()

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

    async def test_extend_accessor_with_super(self):
        motor = await OffsetLinearMotor(position=2 * q.mm)
        self.assertEqual(await motor.get_position(), 3 * q.mm)


class TestContinuousLinearMotor(TestCase):

    async def asyncSetUp(self):
        await super(TestContinuousLinearMotor, self).asyncSetUp()
        self.motor = await ContinuousLinearMotor()

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

    async def test_extend_accessor_with_super_through_diamond(self):
        motor = await OffsetContinuousLinearMotor(position=2 * q.mm)
        self.assertEqual(await motor.get_position(), 3 * q.mm)

    async def test_constructor_arguments_survive_cooperative_initialization(self):
        motor = await ContinuousLinearMotor(
            position=2 * q.mm,
            lower_hard_limit=-3 * q.mm,
            upper_hard_limit=4 * q.mm,
        )
        self.assertEqual(await motor.get_position(), 2 * q.mm)
        self.assertEqual(motor._lower_hard_limit, -3 * q.mm)
        self.assertEqual(motor._upper_hard_limit, 4 * q.mm)

    async def test_cooperative_initializers_run_once(self):
        initializers = (
            ("dummy linear", dummy.LinearMotor),
            ("dummy position mixin", dummy._PositionMixin),
            ("base continuous linear", base.ContinuousLinearMotor),
            ("base linear", base.LinearMotor),
            ("base position mixin", base._PositionMixin),
            ("device", Device),
            ("parameterizable", Parameterizable),
        )
        calls = Counter()

        with ExitStack() as stack:
            for name, cls in initializers:
                original = cls.__ainit__

                async def count_call(self, *args, _name=name, _original=original, **kwargs):
                    calls[_name] += 1
                    await _original(self, *args, **kwargs)

                stack.enter_context(patch.object(cls, "__ainit__", count_call))

            await ContinuousLinearMotor()

        self.assertEqual(calls, Counter(name for name, _ in initializers))


class TestRotationMotor(TestCase):

    async def asyncSetUp(self):
        await super(TestRotationMotor, self).asyncSetUp()
        self.motor = await RotationMotor()

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

    async def asyncSetUp(self):
        await super(TestContinuousRotationMotor, self).asyncSetUp()
        self.motor = await ContinuousRotationMotor()

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

    async def test_constructor_arguments_survive_cooperative_initialization(self):
        motor = await ContinuousRotationMotor(
            position=2 * q.deg,
            lower_hard_limit=-3 * q.deg,
            upper_hard_limit=4 * q.deg,
        )
        self.assertEqual(await motor.get_position(), 2 * q.deg)
        self.assertEqual(motor._lower_hard_limit, -3 * q.deg)
        self.assertEqual(motor._upper_hard_limit, 4 * q.deg)
