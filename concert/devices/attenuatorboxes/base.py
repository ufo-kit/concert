'''
Attenuator module.
Getter and Setter for filter material and the
current state must be provided by the device implementation.
'''

from concert.base import State, AccessorNotImplementedError
from concert.devices.base import Device


class AttenuatorBox(Device):
    '''Attenuator box base class'''
    state = State(default='standby')

    async def __ainit__(self):
        await super(AttenuatorBox, self).__ainit__()

    async def _set_attenuator(self, att):
        raise AccessorNotImplementedError

    async def _get_attenuator(self):
        raise AccessorNotImplementedError


class AttenuatorBoxError(Exception):
    ''' Attenuator related error'''
    pass
