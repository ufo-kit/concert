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
from logbook import Logger
from concert.base import Parameterizable, Parameter
from concert.quantities import q


LOG = Logger(__name__)


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

    .. py:attribute:: state

        Current state of the device.
    """

    NA = "n/a"

    def __init__(self, parameters=None):
        super(Device, self).__init__(parameters)
        self.add_parameter(Parameter('state', self._get_state,
                                     owner_only=True))
        self._lock = threading.Lock()
        self._states = set([self.NA])
        self._state = self.NA

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()

    def _get_state(self):
        return self._state

    def _set_state(self, state):
        if state in self._states:
            self._state = state
            self['state'].notify()
        else:
            LOG.warn("State {0} unknown.".format(state))


class Calibration(object):

    """Interface to convert between user and device units."""

    def to_user(self, value):
        """Return *value* in user units."""
        raise NotImplementedError

    def to_steps(self, value):
        """Return *value* in device units."""
        raise NotImplementedError


class LinearCalibration(Calibration):

    """A linear calibration maps a number of steps to a real-world unit.

    *steps_per_unit* tells how many steps correspond to some unit,
    *offset_in_steps* by how many steps the device is away from some zero
    point.
    """

    def __init__(self, steps_per_unit, offset_in_steps):
        super(LinearCalibration, self).__init__()
        self._steps_per_unit = steps_per_unit
        self._offset = offset_in_steps

    def to_user(self, value_in_steps):
        return value_in_steps * q.count / self._steps_per_unit - self._offset

    def to_steps(self, value):
        res = (value + self._offset) * self._steps_per_unit
        return res.to_base_units().magnitude
