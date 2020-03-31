'''
Lens changer, like used in optics with the possibility to motorized switch objectives.
Getter and Setter for objective and the
current state must be provided by the device implementation.
'''

from concert.base import State, check, AccessorNotImplementedError
from concert.base import Selection
from concert.devices.base import Device


class LensChanger(Device):
    '''Attenuator box base class'''
    state = State(default='standby')
    objective = Selection([], help='objective')

    def __init__(self):
        super(LensChanger, self).__init__()

    @check(source='standby', target='standby')
    def _set_objective(self, objective):
        self._set_objective_real(objective)

    def _get_objective(self):
        return self._get_objective_real()

    def _set_objective_real(self, att):
        raise AccessorNotImplementedError

    def _get_objective_real(self):
        raise AccessorNotImplementedError


class LensChangerError(Exception):
    ''' LensChanger related error'''
    pass
