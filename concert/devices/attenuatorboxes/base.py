'''
Attenuator module.
Getter and Setter for filter material and the
current state must be provided by the device implementation.
'''
from abc import abstractmethod

from concert.base import State
from concert.devices.base import Device


class AttenuatorBox(Device):
    '''Attenuator box base class'''
    state = State(default='standby')

    async def __ainit__(self, **kwargs):
        await super().__ainit__(**kwargs)

    @abstractmethod
    async def _set_attenuator(self, att):
        ...

    @abstractmethod
    async def _get_attenuator(self):
        ...


class AttenuatorBoxError(Exception):
    ''' Attenuator related error'''
    pass
