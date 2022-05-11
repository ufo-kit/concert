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

    async def __ainit__(self):
        await super().__ainit__()
        self._state = 'standby'

    def change_state(self):
        self._state = 'moved'

    async def _get_state(self):
        return self._state


class TestState(TestCase):

    async def test_bad(self):
        device = await BadDevice()
        with self.assertRaises(FSMError):
            await device['state'].get()

    async def test_real_device(self):
        device = await RealDevice()
        self.assertEqual(await device.get_state(), 'standby')
        device.change_state()
        self.assertEqual(await device.get_state(), 'moved')

    async def test_implicit(self):
        device = await ImplicitSoftwareDevice()
        self.assertEqual(await device.get_state(), 'standby')

    async def test_customized(self):
        device = await CustomizedDevice()
        self.assertEqual(await device.get_state(), 'custom')
