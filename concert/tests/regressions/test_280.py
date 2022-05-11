from concert.tests import TestCase
from concert.base import HardLimitError
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor


class TestIssue280(TestCase):

    async def test_hard_limit_throws_off_fsm(self):
        m = await LinearMotor(upper_hard_limit=3 * q.mm)

        with self.assertRaises(HardLimitError):
            await m.set_position(5 * q.m)

        await m.set_position(2 * q.mm)
