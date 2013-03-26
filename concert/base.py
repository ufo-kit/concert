# -*- coding: utf-8 -*-
"""
A device is an abstraction for a piece of hardware that can be controlled.

The main interface to all devices is a generic setter and getter mechanism
provided by every :class:`ConcertObject`. :meth:`ConcertObject.set` sets a
parameter to value. Additionally, you can specify a *blocking* parameter to
halt execution until the value is actually set on the device::

    axis.set('position', 5.5 * q.mm, blocking=True)

    # This will be set once axis.set() has finished
    camera.set('exposure-time', 12.2 * q.s)

Some devices will provide convenience accessor methods. For example, to set the
position on an axis, you can also use :meth:`.Axis.set_position`.

:meth:`ConcertObject.get` simply returns the current value.
"""

import threading
from concert.events.dispatcher import dispatcher
from threading import Event
from logbook import Logger


log = Logger(__name__)


class UnitError(ValueError):
    """Raised when an operation is passed value with an incompatible unit"""
    pass


class LimitError(Exception):
    """Raised when an operation is passed a value that exceeds a limit"""
    pass


class ReadAccessError(Exception):
    """Raised when user tries to change a parameter that cannot be written"""
    def __init__(self, parameter):
        msg = "Parameter {0} cannot be read".format(parameter)
        super(ReadAccessError, self).__init__(msg)


class WriteAccessError(Exception):
    """Raised when user tries to read a parameter that cannot be read"""
    def __init__(self, parameter):
        msg = "Parameter {0} cannot be written".format(parameter)
        super(WriteAccessError, self).__init__(msg)


class MultiContext(object):
    """Multi context manager to be used in a Python `with` management.

    For example, to use multiple axes safely in one process, all you have to do
    is ::

        with MultiContext([axis1, axis2]):
            axis1.set_position()
            axis2.set_position()

    Original code by João S. O. Bueno licensed under CC-BY-3.0.
    """

    def __init__(self, *args):
        if (len(args) == 1 and
               (hasattr(args[0], "__len__") or
                hasattr(args[0], "__iter__"))):
            self.objs = list(args[0])
        else:
            self.objs = args

    def __enter__(self):
        return tuple(obj.__enter__() for obj in self.objs)

    def __exit__(self, type_, value, traceback):
        return all([obj.__exit__(type_, value, traceback)
                    for obj in self.objs])


def value_compatible(value, unit):
    """Check if a *value* of a quantity is compatible with *unit* and return
    ``True``."""
    try:
        unit + value
        return True
    except ValueError:
        return False


class Parameter(object):
    """A parameter with a *name* and an optional *unit* and *limiter*."""
    def __init__(self, name, fget=None, fset=None,
                 unit=None, limiter=None,
                 doc=None):
        self.name = name
        self._setter = fset
        self._getter = fget
        self._unit = unit
        self._limiter = limiter
        self.__doc__ = doc

    def get(self):
        """Try to read and return the current value.

        If the parameter cannot be read, :py:class:`ReadAccessError` is raised.
        """
        if not self.is_readable():
            raise ReadAccessError(self.name)

        return self._getter()

    def set(self, value):
        """Try to write *value*.

        If the parameter cannot be written, :py:class:`WriteAccessError` is
        raised.
        """
        if not self.is_writable():
            raise WriteAccessError(self.name)

        if self._unit and not value_compatible(value, self._unit):
            msg = "`{0}' can only receive values of unit {1} but got {2}"
            raise UnitError(msg.format(self.name, self._unit, value))

        if self._limiter and not self._limiter(value):
            msg = "{0} for `{1}' is out of range"
            raise LimitError(msg.format(value, self.name))

        self._setter(value)

    def is_readable(self):
        """Return `True` if parameter can be read."""
        return self._getter is not None

    def is_writable(self):
        """Return `True` if parameter can be written."""
        return self._setter is not None


class ConcertObject(object):
    """
    Base class handling parameters manipulation. Events are produced
    when a parameter is set. The class provides functionality for
    listening to messages.

    A :class:`ConcertObject` is iterable and returns its parameters of type
    :class:`Parameter` ::

        for param in device:
            msg = "{0} readable={1}, writable={2}"
            print(msg.format(param.name,
                             param.is_readable(), param.is_writable())

    A :class:`ConcertObject` consists of optional getters, setters, limiters
    and units for named parameters. Implementations first call
    :meth:`__init__` and then :meth:`add_parameter` to add or supplement
    a parameter.
    """
    def __init__(self, parameters=None):
        self._params = {}

        if parameters:
            for parameter in parameters:
                self.add_parameter(parameter)

        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, type, value, traceback):
        self._lock.release()

    def __str__(self):
        params = [(param.name, param.get()) for param in self
                  if param.is_readable()]
        params.sort()
        return '\n'.join("%s = %s" % p for p in params)

    def __iter__(self):
        for param in self._params.values():
            yield param

    def get(self, param):
        """Return the value of parameter *name*."""
        if param not in self._params:
            raise ValueError("{0} is not a parameter".format(param))

        return self._params[param].get()

    def set(self, param, value, blocking=False):
        """Set *param* to *value*.

        When *blocking* is true, execution stops until the hardware parameter
        is set to *value*.
        """
        if param not in self._params:
            raise ValueError("{0} is not a parameter".format(param))

        class_name = self.__class__.__name__
        msg = "{0}: set {1}='{2}' blocking='{3}'"
        log.info(msg.format(class_name, param, value, blocking))

        return self._launch(param, self._params[param].set, (value,), blocking)

    def add_parameter(self, parameter):
        self._params[parameter.name] = parameter
        self._register_message(parameter.name)
        setattr(self, parameter.name.replace('-', '_'), parameter.name)

    def _register_message(self, param):
        """Register a message on which one can wait."""
        if not hasattr(self.__class__, "_MESSAGE_TYPES"):
            setattr(self.__class__, "_MESSAGE_TYPES", set([param]))
        else:
            self.__class__._MESSAGE_TYPES.add(param)

    @property
    def message_types(self):
        if not hasattr(self.__class__, "_MESSAGE_TYPES"):
            # In order not to get error if no parameters were set.
            return set([])
        else:
            return self.__class__._MESSAGE_TYPES

    def send(self, message):
        """Send a *message* tied with this object."""
        dispatcher.send(self, message)

    def subscribe(self, message, callback):
        """Subscribe to a *message* from this object.

        *callback* will be called with this object as the first argument.
        """
        dispatcher.subscribe(self, message, callback)

    def unsubscribe(self, message, callback):
        """
        Unsubscribe from a *message*.

        *callback* is a function which is unsubscribed from a particular
        *message* coming from this object.
        """
        dispatcher.unsubscribe(self, message, callback)

    def _launch(self, param, action, args=(), blocking=False):
        """Launch *action* with *args* with message and event handling after
        the action finishes.

        If *blocking* is ``True``, *action* will be called like an ordinary
        function otherwise a thread will be started. *args* must be a tuple of
        arguments that is then unpacked and passed to *action* at _launch time.
        The *action* itself must be blocking.
        """
        def call_action_and_send():
            """Call action and handle its finish."""
            action(*args)
            self.send(param)

        return launch(call_action_and_send, (), blocking)


def launch(action, args=(), blocking=False):
    """Launch *action* with *args*.

    If *blocking* is ``True``, *action* will be called like an ordinary
    function otherwise a thread will be started. *args* must be a tuple of
    arguments that is then unpacked and passed to *action* at launch time.
    The *action* itself must be blocking.

    *launch* returns immediately with an :class:`threading.Event` object that
    can be used to wait for completion of *action*.
    """
    event = Event()

    def call_action_and_complete(args=()):
        """Call action and handle its finish."""
        action(*args)
        event.set()

    if blocking:
        action(*args)
    else:
        thread = threading.Thread(target=call_action_and_complete,
                                  args=(args,))
        thread.daemon = True
        thread.start()

    return event


def wait(events, timeout=None):
    """Wait until all *events* finished."""
    for event in events:
        event.wait(timeout)
