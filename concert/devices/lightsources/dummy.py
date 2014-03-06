"""Dummy lightsource implementation"""

from concert.quantities import q
from concert.devices.lightsources import base


class LightSource(base.LightSource): 
  
    """A dummy light source"""
             
    def __init__(self):
        super(LightSource, self).__init__()
        
    def _set_intensity(self, voltage):        
        self.send(voltage)
            
    def _get_intensity(self):
        return 1 * q.V