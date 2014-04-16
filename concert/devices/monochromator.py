# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:16:41 2014

@author: francis
"""

from concert.devices.motors.dummy import RotationMotor
from concert.devices.base import Device
from concert.base import Selection
from concert.quantities import q

class Monochromator(Device):
    
    """Rotation of motors based on energy level"""
  
    motor = Selection(list(range(1, 44)))
    #motor2 = Selection(list(range(1, 44)))

    def __init__(self):
        super(Monochromator, self).__init__()
        self._motor1 = RotationMotor()
        self._motor2 = RotationMotor()
    
        self._motor1_energy = 0 
        #self._motor2_energy = 0
        
        
    def _set_motor(self, value):
        
        """Rotate the motors"""
        
        self._motor1_energy = value * q.keV
        self._motor1.position = value * q.deg
        self._motor2.position = value * q.deg

    def _get_motor(self):
        
        """Give chosen energylevel"""
        
        return self._motor1_energy