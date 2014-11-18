from concert.base import TransitionNotAllowed
from concert.quantities import q
from concert.devices.xraytubes.dummy import XRayTube
from concert.tests import TestCase


class TestXrayTube(TestCase):

    def setUp(self):
        self.tube = XRayTube()

    def test_on(self):
        if self.tube.state != 'off':
            self.tube.off().join()

        self.tube.on().join()
        self.assertEqual('on', self.tube.state)
        self.assertRaises(TransitionNotAllowed, self.tube.on().join)

    def test_off(self):
        if self.tube.state != 'on':
            self.tube.on().join()

        self.tube.off().join()
        self.assertEqual('off', self.tube.state)
        self.assertRaises(TransitionNotAllowed, self.tube.off().join)

    def test_power(self):
        self.tube.current = 2 * q.A
        self.tube.voltage = 3 * q.V
        self.assertEqual(self.tube.power, 6 * q.W)
