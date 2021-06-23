from concert.base import State, FSMError
from concert.devices.base import Device
from concert.tests import TestCase


class BadDevice(Device):

    state = State()


class ImplicitSoftwareDevice(Device):

    state = State(default='standby')


class CustomizedDevice(Device):

    async def state_getter(self):
        return 'custom'

    state = State(fget=state_getter)


class RealDevice(Device):

    state = State()

    def __init__(self):
        super(RealDevice, self).__init__()
        self._state = 'standby'

    def change_state(self):
        self._state = 'moved'

    async def _get_state(self):
        return self._state


class TestState(TestCase):

    async def test_bad(self):
        device = BadDevice()
        with self.assertRaises(FSMError):
            await device['state'].get()

    def test_real_device(self):
        device = RealDevice()
        self.assertEqual(device.state, 'standby')
        device.change_state()
        self.assertEqual(device.state, 'moved')

    def test_implicit(self):
        device = ImplicitSoftwareDevice()
        self.assertEqual(device.state, 'standby')

    def test_customized(self):
        device = CustomizedDevice()
        self.assertEqual(device.state, 'custom')
