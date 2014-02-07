from concert.devices.shutters.dummy import Shutter as DummyShutter
from concert.tests import TestCase


class TestDummyShutter(TestCase):

    def setUp(self):
        super(TestDummyShutter, self).setUp()
        self.shutter = DummyShutter()

    def test_open(self):
        self.assertEqual(self.shutter.state, 'open')

    def test_close(self):
        self.shutter.close().join()
        self.assertEqual(self.shutter.state, 'closed')
