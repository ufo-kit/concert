from concert.base import TransitionNotAllowed
from concert.devices.grippers.dummy import Gripper
from concert.tests import TestCase


class TestGripper(TestCase):

    def setUp(self):
        self.gripper = Gripper()

    def test_grip(self):
        if self.gripper.state != 'released':
            self.gripper.release().join()

        self.gripper.grip().join()
        self.assertEqual('gripped', self.gripper.state)
        self.assertRaises(TransitionNotAllowed, self.gripper.grip().join)

    def test_release(self):
        if self.gripper.state != 'gripped':
            self.gripper.grip().join()

        self.gripper.release().join()
        self.assertEqual('released', self.gripper.state)
        self.assertRaises(TransitionNotAllowed, self.gripper.release().join)
