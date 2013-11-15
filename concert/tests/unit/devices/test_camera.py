import random
from concert.tests import TestCase
from concert.quantities import q
from concert.devices.cameras.dummy import Camera


class TestDummyCamera(TestCase):

    def setUp(self):
        super(TestDummyCamera, self).setUp()
        self.camera = Camera()

    def test_grab(self):
        frame = self.camera.grab()
        self.assertIsNotNone(frame)

