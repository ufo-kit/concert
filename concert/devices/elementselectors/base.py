"""
ElementSelector module for setting discrete elements (filters, screens, intensity monitors, ...).
Getter and Setter for element and the
current state must be provided by the device implementation.
"""
from abc import abstractmethod

from concert.base import State
from concert.devices.base import Device


class ElementSelector(Device):

    """ElementSelector base class"""

    state = State(default='standby')

    async def __ainit__(self):
        await super().__ainit__()

    @abstractmethod
    async def _set_element(self, element):
        ...

    @abstractmethod
    async def _get_element(self):
        ...


class ElementSelectorError(Exception):
    """ElementSelector related error"""

    pass
