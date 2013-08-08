"""Motor Controller"""
from concert.devices.base import Device
from concert.base import Parameter


class MotorController(Device):

    """Base class for motor controllers."""

    def __init__(self):
        params = [Parameter("motors", self._get_motors)]
        super(MotorController, self).__init__(params)

    def _get_motors(self):
        raise NotImplementedError
