'''
Lens changer, like used in optics with the possibility to motorized switch objectives.
Getter and Setter for objective and the
current state must be provided by the device implementation.
'''
from abc import abstractmethod

from concert.base import State
from concert.devices.base import Device


class LensChanger(Device):
    '''Lens changer base class.'''
    state = State(default='standby')

    async def __ainit__(self):
        await super(LensChanger, self).__ainit__()

    @abstractmethod
    async def _set_objective(self, objective):
        ...

    async def _get_objective(self):
        ...


class LensChangerError(Exception):
    ''' LensChanger related error'''
    pass
