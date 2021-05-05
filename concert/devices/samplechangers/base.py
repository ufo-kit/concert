"""Sample exchanger is a device that will pick a sample from a sample holder and
transfer it onto another one.
"""
from concert.base import AccessorNotImplementedError
from concert.devices.base import Device


class SampleChanger(Device):

    """ A device that moves samples in and out from the sample holder."""

    def __init__(self):
        super(SampleChanger, self).__init__()

    async def _set_sample(self):
        raise AccessorNotImplementedError

    async def _get_sample(self):
        raise AccessorNotImplementedError
