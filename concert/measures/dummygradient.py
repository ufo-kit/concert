'''
Created on Mar 13, 2013

@author: farago
'''
class DummyGradientMeasure(object):
    def __init__(self, axis, max_gradient_position):
        self.max_gradient_position = max_gradient_position
        self._max_gradient = 100
        self._axis = axis
        
    @property
    def maximum_gradient(self):
        return self._max_gradient
    
    def get_gradient(self):
        # Simple quadratic function with its maximum at a set position.
        position = self._axis.get_position().rescale(
                                 self.max_gradient_position.units).magnitude
        return self._max_gradient -\
                (position - self.max_gradient_position.magnitude)**2