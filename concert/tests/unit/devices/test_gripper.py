from concert.base import TransitionNotAllowed
from concert.devices.grippers.dummy import Gripper
from concert.tests import TestCase


class TestGripper(TestCase):

    async def asyncSetUp(self):
        self.gripper = await Gripper()

    async def test_grip(self):
        if await self.gripper.get_state() != 'released':
            await self.gripper.release()

        await self.gripper.grip()
        self.assertEqual('gripped', await self.gripper.get_state())
        with self.assertRaises(TransitionNotAllowed):
            await self.gripper.grip()

    async def test_release(self):
        if await self.gripper.get_state() != 'gripped':
            await self.gripper.grip()

        await self.gripper.release()
        self.assertEqual('released', await self.gripper.get_state())
        with self.assertRaises(TransitionNotAllowed):
            await self.gripper.release()
