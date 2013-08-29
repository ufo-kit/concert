from concert.devices.shutters.dummy import Shutter as DummyShutter
from concert.tests.base import ConcertTest


class TestDummyShutter(ConcertTest):

    def setUp(self):
        super(TestDummyShutter, self).setUp()
        self.shutter = DummyShutter()

    def test_open(self):
        self.shutter.open().wait()
        self.assertEquals(self.shutter.state, self.shutter.OPEN)

    def test_close(self):
        self.shutter.close().wait()
        self.assertEquals(self.shutter.state, self.shutter.CLOSED)
