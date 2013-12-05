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
import threading
import logging
from concert.base import Parameterizable


LOG = logging.getLogger(__name__)


class Device(Parameterizable):

    """
    A :class:`Device` provides locked access to a real-world device and
    provides a :attr:`state` :class:`.Parameter`.

    A implements the context protocol to provide locking and can be used like
    this ::

        with device:
            # device is locked
            device.parameter = 1 * q.m
            ...

        # device is unlocked again
    """

    def __init__(self):
        # We have to create the lock early on because it will be accessed in
        # any add_parameter calls, especially those in the Parameterizable base
        # class
        self._lock = threading.Lock()
        super(Device, self).__init__()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


