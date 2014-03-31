devices.motors.base import RotationMotor 
from concert.devices.base import Device

class Lightfilter(Device):

    def __init__(self):
        #filtermotor = Selection(list(range(1, 6)))
        super(Lightfilter, self).__init__()
        self._filter_motor = RotationMotor()
    
    def _set_filter(self, value):
        self._filter_slot = value
        self._filter_motor.position = 360.0 / value * q.deg 

    def _get_filter(self):
        return self._filter_slot

filt = Lightfilter