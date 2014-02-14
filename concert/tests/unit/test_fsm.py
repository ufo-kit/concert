import time
from concert.tests import TestCase
from concert.base import transition, StateError, TransitionNotAllowed, State
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

    @transition(source='standby', target='moving')
    def start_moving(self, velocity):
        self.velocity = velocity

    @transition(source='*', target='standby')
    def stop_moving(self):
        self.velocity = STOP_VELOCITY

    @async
    @transition(source='standby', target='standby', immediate='moving')
    def move_some_time(self, velocity, duration):
        self.velocity = velocity
        time.sleep(duration)
        self.velocity = STOP_VELOCITY

    @transition(source=['standby', 'moving'], target='standby')
    def stop_no_matter_what(self):
        self.velocity = STOP_VELOCITY

    def actual_state(self):
        return 'standby' if not self.velocity else 'moving'

    @transition(source='*', target=['standby', 'moving'], check=actual_state)
    def set_velocity(self, velocity):
        self.velocity = velocity

    @transition(source='*', target=['ok', 'error'])
    def make_error(self):
        raise StateError('error')

    @transition(source='error', target='standby')
    def reset(self):
        pass


class BaseDevice(Device):

    state = State(default='standby')

    def __init__(self):
        super(BaseDevice, self).__init__()

    @transition(source='standby', target='in-base')
    def switch_base(self):
        pass

    @transition(source='*', target='standby')
    def reset(self):
        pass


class DerivedDevice(BaseDevice):
    state = State(default='standby')

    def __init__(self):
        super(DerivedDevice, self).__init__()

    @transition(source='standby', target='in-derived')
    def switch_derived(self):
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
