'''
Created on Mar 3, 2013

@author: farago
'''
from device.motion.axis.axis import DiscretelyMovable
import numpy
import time
import eventgenerator
from event import Event
import eventtype


class DummyAxis(DiscretelyMovable):
    def __init__(self):
        super(DummyAxis, self).__init__()
        self.position_limit = -numpy.inf, numpy.inf
        self._position = None
        
    def _set_position_real(self, pos):
        try:
            eventgenerator.fire(Event(eventtype.Motion.START, self))
            time.sleep(numpy.random.uniform(0,1))
        finally:
            eventgenerator.fire(Event(eventtype.Motion.STOP, self))
        if self._position != pos:
            eventgenerator.fire(Event(eventtype.Motion.POSITION_CHANGED,
                                      self, pos))
        self._position = pos
        
    def get_position(self):
        return self._position

    