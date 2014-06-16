import time
from concert.tests import TestCase
from concert.base import (transition, check, StateError, TransitionNotAllowed, State, Parameter,
                          FSMError)
from concert.quantities import q
from concert.async import async
from concert.devices.base import Device


STOP_VELOCITY = 0 * q.mm / q.s
MOVE_VELOCITY = 1 * q.mm / q.s


class SomeDevice(Device):

    state = State(default='standby')

    def __init__(self):
        super(SomeDevice, self).__init__()
        self.velocity = STOP_VELOCITY
        self._error = False

    @check(source='standby', target='moving')
    @transition(target='moving')
    def start_moving(self, velocity):
        self.velocity = velocity

    @check(source='*', target='standby')
    @transition(target='standby')
    def stop_moving(self):
        self.velocity = STOP_VELOCITY

    @async
    @check(source='standby', target='standby')
    @transition(immediate='moving', target='standby')
    def move_some_time(self, velocity, duration):
        self.velocity = velocity
        time.sleep(duration)
        self.velocity = STOP_VELOCITY

    @check(source=['standby', 'moving'], target='standby')
    @transition(target='standby')
    def stop_no_matter_what(self):
        self.velocity = STOP_VELOCITY

    def _get_state(self):
        if self._error:
            return 'error'
        return 'standby' if not self.velocity else 'moving'

    @check(source='*', target=['standby', 'moving'])
    def set_velocity(self, velocity):
        self.velocity = velocity

    @check(source='*', target=['ok', 'error'])
    def make_error(self):
        self._error = True
        raise StateError('error')

    @check(source='error', target='standby')
    @transition(target='standby')
    def reset(self):
        self._error = False


class BaseDevice(Device):

    state = State(default='standby')

    def __init__(self):
        super(BaseDevice, self).__init__()

    @check(source='standby', target='in-base')
    @transition(target='in-base')
    def switch_base(self):
        pass

    @check(source='*', target='standby')
    @transition(target='standby')
    def reset(self):
        pass


class DerivedDevice(BaseDevice):
    state = State(default='standby')

    def __init__(self):
        super(DerivedDevice, self).__init__()

    @check(source='standby', target='in-derived')
    @transition(target='in-derived')
    def switch_derived(self):
        pass


class StatelessDevice(Device):
    foo = Parameter(check=check(source='*', target='*'))

    @transition(target='changed')
    def change(self):
        pass


class TestStateMachine(TestCase):

    def setUp(self):
        super(TestStateMachine, self).setUp()
        self.device = SomeDevice()

    def test_state_per_device(self):
        d1 = SomeDevice()
        d2 = SomeDevice()

        d2.start_moving(MOVE_VELOCITY)
        self.assertNotEqual(d1.state, d2.state)

    def test_defaults(self):
        self.device.state == 'standby'

    def test_valid_transition(self):
        self.device.start_moving(MOVE_VELOCITY)
        self.assertEqual(self.device.state, 'moving')

        self.device.stop_moving()
        self.assertEqual(self.device.state, 'standby')

    def test_async_transition(self):
        future = self.device.move_some_time(MOVE_VELOCITY, 0.01)
        future.join()
        self.assertEqual(self.device.state, 'standby')

    def test_multiple_source_states(self):
        self.device.stop_no_matter_what()
        self.assertEqual(self.device.state, 'standby')

        self.device.start_moving(MOVE_VELOCITY)
        self.device.stop_no_matter_what()

        self.assertEqual(self.device.state, 'standby')

    def test_multiple_target_states(self):
        self.device.set_velocity(MOVE_VELOCITY)
        self.assertEqual(self.device.state, 'moving')

        self.device.set_velocity(STOP_VELOCITY)
        self.assertEqual(self.device.state, 'standby')

    def test_inheritance(self):
        device = DerivedDevice()

        device.switch_base()
        self.assertEqual(device.state, 'in-base')

        with self.assertRaises(TransitionNotAllowed):
            device.switch_base()

        with self.assertRaises(TransitionNotAllowed):
            device.switch_derived()

        device.reset()
        self.assertEqual(device.state, 'standby')

        device.switch_derived()
        self.assertEqual(device.state, 'in-derived')

        with self.assertRaises(TransitionNotAllowed):
            device.switch_derived()

        with self.assertRaises(TransitionNotAllowed):
            device.switch_base()

        device.reset()
        self.assertEqual(device.state, 'standby')

    def test_errors(self):
        with self.assertRaises(StateError):
            self.device.make_error()

        self.assertEqual(self.device.state, 'error')
        self.device.reset()
        self.assertEqual(self.device.state, 'standby')

    def test_state_setting(self):
        with self.assertRaises(AttributeError):
            self.device.state = 'foo'

    def test_stateless_parameter_transition(self):
        dev = StatelessDevice()
        with self.assertRaises(FSMError):
            dev.foo = 1

    def test_stateless_transition(self):
        dev = StatelessDevice()
        with self.assertRaises(FSMError):
            dev.change()

    def test_no_default_state(self):
        class BadStatelessDevice(Device):
            state = State()

        with self.assertRaises(FSMError):
            BadStatelessDevice().state
