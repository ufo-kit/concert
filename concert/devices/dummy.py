'''
Created on Mar 14, 2013

@author: farago
'''
import random
import time
from concert.base import ConcertObject


class DummyDevice(ConcertObject):
    """A dummy device."""
    def __init__(self):
        super(DummyDevice, self).__init__()
        self._register("value", self._get_value, self._set_value)
        self._value = None

    def _get_value(self):
        """Get the real value."""
        return self._value

    def _set_value(self, value):
        """The real value setter."""
        time.sleep(random.random()/50.)
        self._value = value

    def get_value(self):
        """Get value."""
        return self.get("value")

    def set_value(self, value):
        """Set *value*."""
        return self.set("value", value)
