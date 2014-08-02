from concert.tests import TestCase
from concert.base import HardLimitError
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor


class TestIssue280(TestCase):

    def test_hard_limit_throws_off_fsm(self):
        m = LinearMotor()

        with self.assertRaises(HardLimitError):
            m.position = 5 * q.m

        m.position = 2 * q.mm
