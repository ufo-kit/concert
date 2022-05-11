from concert.tests import TestCase
from concert.base import TransitionNotAllowed
from concert.devices.io.base import IODeviceError
from concert.devices.io.dummy import IO, Signal


class TestIO(TestCase):

    async def asyncSetUp(self):
        self.io = await IO(port_value=0)
        self.port = 0

    async def test_read(self):
        self.assertEqual(0, await self.io.read_port(self.port))

    async def test_write(self):
        value = 1
        await self.io.write_port(self.port, value)
        self.assertEqual(value, await self.io.read_port(self.port))

    async def test_non_existent_read(self):
        with self.assertRaises(IODeviceError):
            await self.io.read_port(1)

    async def test_non_existent_write(self):
        with self.assertRaises(IODeviceError):
            await self.io.write_port(1, 0)


class TestSignal(TestCase):

    async def asyncSetUp(self):
        self.signal = await Signal()

    async def test_on(self):
        await self.signal.on()
        self.assertEqual(await self.signal.get_state(), 'on')

        with self.assertRaises(TransitionNotAllowed):
            await self.signal.on()

    async def test_off(self):
        await self.signal.on()
        await self.signal.off()
        self.assertEqual(await self.signal.get_state(), 'off')

        with self.assertRaises(TransitionNotAllowed):
            await self.signal.off()

    async def test_trigger(self):
        await self.signal.trigger()
        self.assertEqual(await self.signal.get_state(), 'off')

        # Test forbidden state
        await self.signal.on()
        with self.assertRaises(TransitionNotAllowed):
            await self.signal.trigger()
