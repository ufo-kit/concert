"""Sample exchanger is a device that will pick a sample from a sample holder and
transfer it onto another one.
"""
from abc import abstractmethod
from concert.devices.base import Device


class SampleChanger(Device):

    """ A device that moves samples in and out from the sample holder."""

    async def __ainit__(self):
        await super(SampleChanger, self).__ainit__()

    @abstractmethod
    async def _set_sample(self):
        ...

    @abstractmethod
    async def _get_sample(self):
        ...
