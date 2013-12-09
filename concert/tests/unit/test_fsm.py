import time
from concert.tests import TestCase
from concert.quantities import q
from concert.fsm import transition, State
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

    @transition(source='*')
    def cause_erroneous_behaviour(self, msg):
        raise Exception(msg)


class TestStateMachine(TestCase):

    def setUp(self):
        super(TestStateMachine, self).setUp()
        self.device = SomeDevice()

    def test_state_per_device(self):
        d1 = SomeDevice()
        d2 = SomeDevice()

        d2.start_moving(MOVE_VELOCITY)
        self.assertNotEqual(d1.state.is_currently('moving'),
                            d2.state.is_currently('moving'))

    def test_defaults(self):
        self.device.state.value == 'standby'
        self.device.state.is_currently('standby')

    def test_valid_transition(self):
        self.device.start_moving(MOVE_VELOCITY)
        self.assertTrue(self.device.state.is_currently('moving'))

        self.device.stop_moving()
        self.assertTrue(self.device.state.is_currently('standby'))

    def test_async_transition(self):
        future = self.device.move_some_time(MOVE_VELOCITY, 0.01)
        future.join()
        self.assertTrue(self.device.state.is_currently('standby'))

    def test_multiple_source_states(self):
        self.device.stop_no_matter_what()
        self.assertTrue(self.device.state.is_currently('standby'))

        self.device.start_moving(MOVE_VELOCITY)
        self.device.stop_no_matter_what()

        self.assertTrue(self.device.state.is_currently('standby'))

    def test_error(self):
        with self.assertRaises(Exception):
            self.device.cause_erroneous_behaviour("Oops")

        self.assertTrue(self.device.state.is_currently('error'))
        self.assertEqual(self.device.state.error, "Oops")
