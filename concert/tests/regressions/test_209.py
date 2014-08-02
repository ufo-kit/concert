from concert.base import FSMError
from concert.tests import TestCase
from concert.quantities import q
from concert.devices.motors.base import RotationMotor as BaseRotationMotor
from concert.devices.motors.dummy import RotationMotor


class ImproperlyImplemented(BaseRotationMotor):
    def __init__(self):
        super(ImproperlyImplemented, self).__init__()
        self._value = 0

    def _get_position(self):
        return self._value

    def _set_position(self, value):
        self._value = value


class BreakingMotor(ImproperlyImplemented):
    def __init__(self):
        super(BreakingMotor, self).__init__()

    def _get_state(self):
        return 'standby'


class TestIssue209(TestCase):

    def test_method_not_implemented(self):
        """
        Although required by base.RotationMotor, the BreakingMotor does not
        implement _get_state but rather than telling us that, it says that the
        parameter cannot be written.
        """
        fancy = ImproperlyImplemented()

        with self.assertRaises(FSMError):
            fancy.position = 20 * q.deg

    def test_shared_parameters(self):
        dummy = RotationMotor()
        fancy = BreakingMotor()

        dummy.position = 10 * q.deg
        self.assertEqual(dummy.position, 10 * q.deg)

        fancy.position = 20 * q.deg
        self.assertEqual(fancy.position, 20 * q.deg)
