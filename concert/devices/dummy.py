'''
Created on Mar 14, 2013

@author: farago
'''
import random
import time
from concert.base import Device, Parameter, AsyncWrapper
from concurrent import futures


class DummyDevice(Device):
    """A dummy device."""
    def __init__(self):
        parameter = Parameter('value', self._get_value, self._set_value)
        super(DummyDevice, self).__init__([parameter])
        self._value = None

    def _get_value(self):
        """Get the real value."""
        return self._value

    def _set_value(self, value):
        """The real value setter."""
        time.sleep(random.random())
        self._value = value

    def do_nothing(self):
        time.sleep(random.random())


if __name__ == '__main__':
    # Property.
    ad = AsyncWrapper(DummyDevice())
    ad.value = 12
    print ad.value

    # Parameter.
    future = ad.set_value(180)
    futures.wait([future])
    print ad.get_value().result()

    print "Asynchronous operation."
    future = ad.do_nothing()
    print "Waiting for asynchronous operation to finish."
    futures.wait([future])
    print "Done."
