import time
from concert.tests import TestCase
from concert.base import State
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

    @state.transition(source='standby', target='moving')
    def start_moving(self, velocity):
        self.velocity = velocity

    @state.transition(source='*', target='standby')
    def stop_moving(self):
        self.velocity = STOP_VELOCITY

    @async
    @state.transition(source='standby', target='standby', immediate='moving')
    def move_some_time(self, velocity, duration):
        self.velocity = velocity
        time.sleep(duration)
        self.velocity = STOP_VELOCITY

    @state.transition(source=['standby', 'moving'], target='standby')
    def stop_no_matter_what(self):
        self.velocity = STOP_VELOCITY

    def actual_state(self):
        return 'standby' if not self.velocity else 'moving'

    @state.transition(source='*', target=['standby', 'moving'], check=actual_state)
    def set_velocity(self, velocity):
        self.velocity = velocity

    def reset(self):
        self.state.reset()


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
