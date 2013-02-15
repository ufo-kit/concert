import logging

class Maximizer(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        self._last_value = -1000000.0
        
    @property
    def last_value(self):
        return self._last_value
    
    @last_value.setter
    def last_value(self, val):
        self._last_value = val

    def is_better(self, value):
        return self._last_value < value

    def set_point_reached(self, value):
        return abs(self._last_value - value) < 0.01
