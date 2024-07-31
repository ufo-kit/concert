"""
Photo diode device base class
"""
from abc import abstractmethod

from concert.quantities import q
from concert.base import Quantity
from concert.devices.base import Device


class PhotoDiode(Device):

    """
    Impementation of photo diode with V output signal
    """

    intensity = Quantity(q.V)

    async def __ainit__(self):
        await super(PhotoDiode, self).__ainit__()

    @abstractmethod
    async def _get_intensity(self):
        ...
