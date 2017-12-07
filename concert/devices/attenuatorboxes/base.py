'''
Attenuator module.
Getter and Setter for filter material and the
current state must be provided by the device implementation.
'''

from concert.base import State, check, AccessorNotImplementedError
from concert.base import Selection
from concert.devices.base import Device


class AttenuatorBox(Device):
    '''Attenuator box base class'''
    state = State(default='standby')
    attenuator = Selection([], help='Filter material')

    def __init__(self):
        super(AttenuatorBox, self).__init__()

    @check(source='standby', target='standby')
    def _set_attenuator(self, att):
        self._set_attenuator_real(att)

    def _get_attenuator(self):
        return self._get_attenuator_real()

    def _set_attenuator_real(self, att):
        raise AccessorNotImplementedError

    def _get_attenuator_real(self):
        raise AccessorNotImplementedError


class AttenuatorBoxError(Exception):
    ''' Attenuator related error'''
    pass
