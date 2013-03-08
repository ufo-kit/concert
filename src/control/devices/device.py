'''
Created on Mar 6, 2013

@author: farago
'''
import itertools
from control.controlobject import Identifiable


make_state_id = itertools.count().next


class UnknownStateError(Exception):
    """Any limit (hard or soft) exception."""
    def __init__(self, message):
        self._message = message

    def __str__(self):
        return repr(self._message)


class State(object):
    """Status of a device.

    This is NOT a connection status, but a reflection of a physical device
    status. The implementation should follow this guideline.

    """
    ERROR = make_state_id()


class Device(Identifiable):
    """A device with a state."""
    def __init__(self):
        super(Device, self).__init__()

    @property
    def state(self):
        raise NotImplementedError
