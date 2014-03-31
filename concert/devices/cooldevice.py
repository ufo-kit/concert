"""Filter selection"""
from concert.devices.motors.dummy import RotationMotor
from concert.devices.base import Device
from concert.base import Selection
from concert.quantities import q


class Lightfilter(Device):
    
    """Filter selection motor"""
  
    filtermotor = Selection(list(range(1, 6)))

    def __init__(self):
        super(Lightfilter, self).__init__()
        self._filter_select = RotationMotor()
        self._filter_slot = 0

    def _set_filter(self, value):
        """Set the chosen filter"""
        
        self._filter_slot = value
        self._filter_select.position = 360.0 / value * q.deg

    def _get_filter(self):
        """Give chosen filter"""
        
        return self._filter_slot
