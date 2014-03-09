"""Light sources"""
from concert.devices.base import Device
from concert.base import AccessorNotImplementedError

class LightSource(Device):

    """A base LightSource class."""
    
    def __init__(self):        
        super(LightSource, self).__init__()        
        
    def _set_voltage(self, value):               
        raise AccessorNotImplementedError
    
    def _get_voltage(self):
        raise AccessorNotImplementedError
        
    def set_on(self):
        raise AccessorNotImplementedError
        
    def set_off(self):
        raise AccessorNotImplementedError
