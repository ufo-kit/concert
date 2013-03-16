"""
A device is an abstraction for a piece of hardware that can be controlled.

The main interface to all devices is a generic setter and getter mechanism.
:meth:`Device.set` sets a parameter to value. Additionally, you can specify a
*blocking* parameter to halt execution until the value is actually set on the
device::

    axis.set('position', 5.5 * q.mm, blocking=True)

    # This will be set once axis.set() has finished
    camera.set('exposure-time', 12.2 * q.s)

Some devices will provide convenience accessor methods. For example, to set the
position on an axis, you can also use :meth:`.Axis.set_position`.

:meth:`Device.get` simply returns the current value.
"""

from logbook import Logger


log = Logger(__name__)


class UnknownStateError(Exception):
    """Any limit (hard or soft) exception."""
    def __init__(self, message):
        self._message = message

    def __str__(self):
        return repr(self._message)


class State(object):
    """State of a device.

    This is NOT a connection status, but a reflection of a physical device
    status. The implementation should follow this guideline.

    """
    ERROR = "error"
