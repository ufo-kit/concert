from concert.devices.shutters.dummy import Shutter as DummyShutter
from concert.tests import TestCase


class TestDummyShutter(TestCase):

    async def asyncSetUp(self):
        await super(TestDummyShutter, self).asyncSetUp()
        self.shutter = await DummyShutter()

    async def test_open(self):
        self.assertEqual(await self.shutter.get_state(), 'open')

    async def test_close(self):
        await self.shutter.close()
        self.assertEqual(await self.shutter.get_state(), 'closed')
