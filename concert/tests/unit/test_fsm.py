import asyncio
import inspect
from concert.tests import TestCase
from concert.base import (transition, check, StateError, TransitionNotAllowed, State, Parameter,
                          FSMError)
from concert.quantities import q
from concert.devices.base import Device


STOP_VELOCITY = 0 * q.mm / q.s
MOVE_VELOCITY = 1 * q.mm / q.s


class SomeDevice(Device):

    state = State(default='standby')

    async def __ainit__(self):
        await super(SomeDevice, self).__ainit__()
        self.velocity = STOP_VELOCITY
        self.faulty = False
        self._error = False

    @transition(target='standby')
    async def make_transition(self):
        if self.faulty:
            raise RuntimeError

    @check(source='standby', target='moving')
    @transition(target='moving')
    async def start_moving(self, velocity):
        self.velocity = velocity

    @check(source='*', target='standby')
    @transition(target='standby')
    async def stop_moving(self):
        self.velocity = STOP_VELOCITY

    @check(source='standby', target='standby')
    @transition(immediate='moving', target='standby')
    async def move_some_time(self, velocity, duration):
        self.velocity = velocity
        await asyncio.sleep(duration)
        self.velocity = STOP_VELOCITY

    @check(source=['standby', 'moving'], target='standby')
    @transition(target='standby')
    async def stop_no_matter_what(self):
        self.velocity = STOP_VELOCITY

    async def _get_state(self):
        if self.faulty:
            raise RuntimeError
        if self._error:
            return 'error'
        return 'standby' if not self.velocity else 'moving'

    @check(source='*', target=['standby', 'moving'])
    async def set_velocity(self, velocity):
        self.velocity = velocity

    @check(source='*', target=['ok', 'error'])
    async def make_error(self):
        self._error = True
        raise StateError('error')

    @check(source='error', target='standby')
    @transition(target='standby')
    async def reset(self):
        self._error = False

    @check()
    async def produce(self):
        yield None


class BaseDevice(Device):

    state = State(default='standby')

    async def __ainit__(self):
        await super(BaseDevice, self).__ainit__()

    @check(source='standby', target='in-base')
    @transition(target='in-base')
    async def switch_base(self):
        pass

    @check(source='*', target='standby')
    @transition(target='standby')
    async def reset(self):
        pass


class DerivedDevice(BaseDevice):
    state = State(default='standby')

    async def __ainit__(self):
        await super(DerivedDevice, self).__ainit__()

    @check(source='standby', target='in-derived')
    @transition(target='in-derived')
    async def switch_derived(self):
        pass


class StatelessDevice(Device):
    foo = Parameter(check=check(source='*', target='*'))

    @transition(target='changed')
    async def change(self):
        pass


class MultipleStateDevice(Device):
    foo_state = State(default='standby')
    bar_state = State()
    state = State(default='standby')

    parameter = Parameter(check=check(source='standby', state_name='foo_state'))

    async def __ainit__(self):
        self._bar_state = "standby"
        await super().__ainit__()

    @check(source='standby', target='moving')
    @transition(target='moving')
    async def switch_state_to_moving(self):
        pass

    @check(source='moving', target='standby')
    @transition(target='standby')
    async def switch_state_to_standby(self):
        pass

    @check(source='standby', target='moving', state_name='foo_state')
    @transition(target='moving', state_name='foo_state')
    async def switch_foo_state_to_moving(self):
        pass

    @check(source='moving', target='standby', state_name='foo_state')
    @transition(target='standby', state_name='foo_state')
    async def switch_foo_state_to_standby(self):
        pass

    async def _get_bar_state(self):
        return self._bar_state

    @check(source='standby', target='moving', state_name='bar_state')
    async def switch_bar_state_to_moving(self):
        self._bar_state = "moving"

    @check(source='moving', target='standby', state_name='bar_state')
    async def switch_bar_state_to_standby(self):
        self._bar_state = "standby"

    async def _get_parameter(self):
        return True

    async def _set_parameter(self, value):
        pass


class TestStateMachine(TestCase):

    async def asyncSetUp(self):
        await super(TestStateMachine, self).asyncSetUp()
        self.device = await SomeDevice()

    async def test_state_per_device(self):
        d1 = await SomeDevice()
        d2 = await SomeDevice()

        await d2.start_moving(MOVE_VELOCITY)
        self.assertNotEqual(await d1.get_state(), await d2.get_state())

    async def test_defaults(self):
        await self.device.get_state() == 'standby'

    async def test_valid_transition(self):
        await self.device.start_moving(MOVE_VELOCITY)
        self.assertEqual(await self.device.get_state(), 'moving')

        await self.device.stop_moving()
        self.assertEqual(await self.device.get_state(), 'standby')

    async def test_casync_transition(self):
        await self.device.move_some_time(MOVE_VELOCITY, 0.01)
        self.assertEqual(await self.device.get_state(), 'standby')

    async def test_multiple_source_states(self):
        await self.device.stop_no_matter_what()
        self.assertEqual(await self.device.get_state(), 'standby')

        await self.device.start_moving(MOVE_VELOCITY)
        await self.device.stop_no_matter_what()

        self.assertEqual(await self.device.get_state(), 'standby')

    async def test_multiple_target_states(self):
        await self.device.set_velocity(MOVE_VELOCITY)
        self.assertEqual(await self.device.get_state(), 'moving')

        await self.device.set_velocity(STOP_VELOCITY)
        self.assertEqual(await self.device.get_state(), 'standby')

    async def test_inheritance(self):
        device = await DerivedDevice()

        await device.switch_base()
        self.assertEqual(await device.get_state(), 'in-base')

        with self.assertRaises(TransitionNotAllowed):
            await device.switch_base()

        with self.assertRaises(TransitionNotAllowed):
            await device.switch_derived()

        await device.reset()
        self.assertEqual(await device.get_state(), 'standby')

        await device.switch_derived()
        self.assertEqual(await device.get_state(), 'in-derived')

        with self.assertRaises(TransitionNotAllowed):
            await device.switch_derived()

        with self.assertRaises(TransitionNotAllowed):
            await device.switch_base()

        await device.reset()
        self.assertEqual(await device.get_state(), 'standby')

    async def test_errors(self):
        with self.assertRaises(StateError):
            await self.device.make_error()

        self.assertEqual(await self.device.get_state(), 'error')
        await self.device.reset()
        self.assertEqual(await self.device.get_state(), 'standby')

    async def test_state_setting(self):
        with self.assertRaises(AttributeError):
            self.device.state = 'foo'

    async def test_stateless_parameter_transition(self):
        dev = await StatelessDevice()
        with self.assertRaises(FSMError):
            await dev.set_foo(1)

    async def test_stateless_transition(self):
        dev = await StatelessDevice()
        with self.assertRaises(FSMError):
            await dev.change()

    async def test_transition_exception(self):
        dev = await SomeDevice()
        dev.faulty = True
        with self.assertRaises(RuntimeError):
            await dev.make_transition()

    async def test_check_exception(self):
        dev = await SomeDevice()
        dev.faulty = True
        with self.assertRaises(RuntimeError):
            await dev.start_moving()

    async def test_check_coro(self):
        self.assertTrue(inspect.iscoroutinefunction(self.device.start_moving))

    async def test_check_asyncgen(self):
        self.assertTrue(inspect.isasyncgenfunction(self.device.produce))

    async def test_multiple_state(self):
        dev = await MultipleStateDevice()
        self.assertEqual(await dev.get_state(), 'standby')
        self.assertEqual(await dev.get_foo_state(), 'standby')
        self.assertEqual(await dev.get_bar_state(), 'standby')
        await dev.set_parameter(False)

        await dev.switch_state_to_moving()
        self.assertEqual(await dev.get_state(), 'moving')
        self.assertEqual(await dev.get_foo_state(), 'standby')
        self.assertEqual(await dev.get_bar_state(), 'standby')
        await dev.set_parameter(False)

        await dev.switch_foo_state_to_moving()
        self.assertEqual(await dev.get_state(), 'moving')
        self.assertEqual(await dev.get_foo_state(), 'moving')
        self.assertEqual(await dev.get_bar_state(), 'standby')
        with self.assertRaises(FSMError):
            await dev.set_parameter(True)

        await dev.switch_bar_state_to_moving()
        self.assertEqual(await dev.get_state(), 'moving')
        self.assertEqual(await dev.get_foo_state(), 'moving')
        self.assertEqual(await dev.get_bar_state(), 'moving')
        with self.assertRaises(FSMError):
            await dev.set_parameter(True)

        await dev.switch_state_to_standby()
        self.assertEqual(await dev.get_state(), 'standby')
        self.assertEqual(await dev.get_foo_state(), 'moving')
        self.assertEqual(await dev.get_bar_state(), 'moving')
        with self.assertRaises(FSMError):
            await dev.set_parameter(True)

        await dev.switch_foo_state_to_standby()
        self.assertEqual(await dev.get_state(), 'standby')
        self.assertEqual(await dev.get_foo_state(), 'standby')
        self.assertEqual(await dev.get_bar_state(), 'moving')
        await dev.set_parameter(False)

        await dev.switch_bar_state_to_standby()
        self.assertEqual(await dev.get_state(), 'standby')
        self.assertEqual(await dev.get_foo_state(), 'standby')
        self.assertEqual(await dev.get_bar_state(), 'standby')
        await dev.set_parameter(False)
