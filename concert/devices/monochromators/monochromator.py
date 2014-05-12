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
        self._motor1 = RotationMotor()
        self._motor2 = RotationMotor()

    def _set_wavelength(self, value):

        """Define wavelength and rotate motors"""

        self._motor1.position = value.magnitude * q.deg
        self._motor2.position = value.magnitude * q.deg

    def _get_wavelength(self):

        """Obtain the wavelength"""

        return self._motor1.position.magnitude * q.m