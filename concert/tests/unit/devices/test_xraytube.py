from concert.base import TransitionNotAllowed
from concert.quantities import q
from concert.devices.xraytubes.dummy import XRayTube
from concert.tests import TestCase


class TestXrayTube(TestCase):

    async def asyncSetUp(self):
        self.tube = await XRayTube()

    async def test_on(self):
        if await self.tube.get_state() != 'off':
            await self.tube.off()

        await self.tube.on()
        self.assertEqual('on', await self.tube.get_state())
        with self.assertRaises(TransitionNotAllowed):
            await self.tube.on()

    async def test_off(self):
        if await self.tube.get_state() != 'on':
            await self.tube.on()

        await self.tube.off()
        self.assertEqual('off', await self.tube.get_state())
        with self.assertRaises(TransitionNotAllowed):
            await self.tube.off()

    def test_power(self):
        self.tube.current = 2 * q.A
        self.tube.voltage = 3 * q.V
        self.assertEqual(self.tube.power, 6 * q.W)
