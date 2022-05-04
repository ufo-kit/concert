'''
Lens changer, like used in optics with the possibility to motorized switch objectives.
Getter and Setter for objective and the
current state must be provided by the device implementation.
'''

from concert.base import State, AccessorNotImplementedError
from concert.devices.base import Device


class LensChanger(Device):
    '''Lens changer base class.'''
    state = State(default='standby')

    async def __ainit__(self):
        await super(LensChanger, self).__ainit__()

    async def _set_objective(self, objective):
        raise AccessorNotImplementedError

    async def _get_objective(self):
        raise AccessorNotImplementedError


class LensChangerError(Exception):
    ''' LensChanger related error'''
    pass
