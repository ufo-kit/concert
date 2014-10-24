"""Sample exchanger is a device that will pick a sample from a sample holder and
transfer it onto another one.
"""
from concert.base import Selection, AccessorNotImplementedError
from concert.devices.base import Device


class SampleChanger(Device):

    """ A device that moves samples in and out from the sample holder."""

    sample = Selection([None])

    def __init__(self):
        super(SampleChanger, self).__init__()

    def _set_sample(self):
        raise AccessorNotImplementedError

    def _get_sample(self):
        raise AccessorNotImplementedError
