"""
ElementSelector module for setting discrete elements (filters, screens, intensity monitors, ...).
Getter and Setter for element and the
current state must be provided by the device implementation.
"""

from concert.base import State, AccessorNotImplementedError
from concert.devices.base import Device


class ElementSelector(Device):

    """ElementSelector base class"""

    state = State(default='standby')

    async def __ainit__(self):
        await super().__ainit__()

    async def _set_element(self, element):
        raise AccessorNotImplementedError

    async def _get_element(self):
        raise AccessorNotImplementedError


class ElementSelectorError(Exception):
    """ElementSelector related error"""

    pass
