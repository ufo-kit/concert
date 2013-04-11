'''
Created on Apr 11, 2013

@author: farago
'''
from concert.base import Parameterizable, Parameter


class MotorController(Parameterizable):
    """Base class for motor controllers."""
    def __init__(self):
        params = [Parameter("motors", self._get_motors)]
        super(MotorController, self).__init__(params)
        
    def _get_motors(self):
        raise NotImplementedError