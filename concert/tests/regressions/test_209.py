from concert.quantities import q
from concert.devices.motors.base import RotationMotor as BaseRotationMotor
from concert.devices.motors.dummy import RotationMotor
 
 
class BreakingMotor(BaseRotationMotor):
    def __init__(self):
        super(BreakingMotor, self).__init__()
        self._value = 0
     
    def _get_position(self):
        return self._value
     
    def _set_position(self, value):
        self._value = value

    def check_state(self):
        return 'standby'
 
 
def test_issue_209():
    dummy = RotationMotor()
    fancy = BreakingMotor()

    dummy.position = 10 * q.deg
    assert dummy.position == 10 * q.deg

    fancy.position = 20 * q.deg
    assert fancy.position == 20 * q.deg
