from concert.devices.shutters.dummy import Shutter as DummyShutter
from concert.tests import TestCase


class TestDummyShutter(TestCase):

    def setUp(self):
        super(TestDummyShutter, self).setUp()
        self.shutter = DummyShutter()

    async def test_open(self):
        self.assertEqual(await self.shutter.get_state(), 'open')

    async def test_close(self):
        await self.shutter.close()
        self.assertEqual(await self.shutter.get_state(), 'closed')
