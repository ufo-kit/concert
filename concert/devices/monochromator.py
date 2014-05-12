# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:16:41 2014

@author: francis
"""

from concert.devices.motors.dummy import RotationMotor
from concert.devices.base import Device
from concert.quantities import q
from concert.base import Quantity

class Monochromator(Device):
    
    """Rotation of motors based on wavelength"""
    
    wavelength = Quantity(q.m)

    def __init__(self):
        super(Monochromator, self).__init__()
        #self.wavelength = None * q.m
        self._motor1 = RotationMotor()
        self._motor2 = RotationMotor()
    
    def set_wavelength(self, value):
        
        """Rotate the motors"""
        self.wavelength = value
        self.magnitude = str(self.wavelength)
        self._motor1.position = (float(self.magnitude.split(" ")[0]) * 1000) * q.deg
        self._motor2.position = (float(self.magnitude.split(" ")[0]) * 1000) * q.deg
  
    def get_wavelength(self):
        
        """Give chosen energylevel"""
        
        return self.wavelength