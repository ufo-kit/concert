from concert.tests import TestCase
from concert.quantities import q
from concert.devices.motors.base import RotationMotor as BaseRotationMotor
from concert.devices.motors.dummy import RotationMotor


class ImproperlyImplemented(BaseRotationMotor):
    async def __ainit__(self):
        await super(ImproperlyImplemented, self).__ainit__()
        self._value = 0

    async def _get_position(self):
        return self._value

    async def _set_position(self, value):
        self._value = value


class BreakingMotor(ImproperlyImplemented):
    async def __ainit__(self):
        await super(BreakingMotor, self).__ainit__()

    async def _get_state(self):
        return 'standby'


class TestIssue209(TestCase):

    async def test_method_not_implemented(self):
        """
        Although required by base.RotationMotor, the BreakingMotor does not
        implement _get_state but rather than telling us that, it says that the
        parameter cannot be written.
        """
        fancy = await ImproperlyImplemented()

        with self.assertRaises(NotImplementedError):
            await fancy.set_position(20 * q.deg)

    async def test_shared_parameters(self):
        dummy = await RotationMotor()
        fancy = await BreakingMotor()

        await dummy.set_position(10 * q.deg)
        self.assertEqual(await dummy.get_position(), 10 * q.deg)

        await fancy.set_position(20 * q.deg)
        self.assertEqual(await fancy.get_position(), 20 * q.deg)
