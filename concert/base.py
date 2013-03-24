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


class LimitError(ValueError):
    """Raised when an operation is passed a value that exceeds a limit"""
    pass


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


class ConcertObject(object):
    """
    Base class handling parameters manipulation. Events are produced
    when a parameter is set. The class provides functionality for
    listening to messages.

    A :class:`ConcertObject` consists of optional getters, setters, limiters
    and units for named parameters. Implementations first call
    :meth:`__init__` and then :meth:`_register` to add or supplement
    a parameter.
    """
    def __init__(self):
        self._setters = {}
        self._getters = {}
        self._limiters = {}
        self._units = {}
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, type, value, traceback):
        self._lock.release()

    def __str__(self):
        s = str(self.__class__) + " "
        for param, get in self._getters.iteritems():
            s += "{0} = {1}".format(param, get())
        return s

    def get(self, param):
        """Return the value of *param*."""
        if param not in self._getters:
            raise NotImplementedError

        return self._getters[param]()

    def set(self, param, value, blocking=False):
        """Set *param* to *value*.

        When *blocking* is true, execution stops until the hardware parameter
        is set to *value*.
        """
        if param not in self._setters:
            raise NotImplementedError

        if not self._unit_is_compatible(param, value):
            msg = "`{0}' can only receive values of unit {1}"
            raise UnitError(msg.format(param, self._units[param]))

        if not self._value_is_in_range(param, value):
            msg = "{0} for `{1}` is out of range"
            raise LimitError(msg.format(value, param))

        class_name = self.__class__.__name__
        msg = "{0}: set {1}='{2}' blocking='{3}'"
        log.info(msg.format(class_name, param, value, blocking))

        setter = self._setters[param]
        return self._launch(param, setter, (value,), blocking)

    def _unit_is_compatible(self, param, value):
        if not param in self._units:
            return True

        try:
            self._units[param] + value
            return True
        except ValueError:
            return False

    def _value_is_in_range(self, param, value):
        if param in self._limiters:
            return self._limiters[param](value)

        return True

    def _register_getter(self, param, getter):
        if param in self._getters:
            transform = self._getters[param]
            self._getters[param] = lambda: transform(getter())
        else:
            self._getters[param] = getter

    def _register_setter(self, param, setter):
        if param in self._setters:
            transform = self._setters[param]
            self._setters[param] = lambda arg: setter(transform(arg))
        else:
            self._setters[param] = setter

    def _register_message(self, param):
        """Register a message on which one can wait."""
        if not hasattr(self.__class__, "_MESSAGE_TYPES"):
            setattr(self.__class__, "_MESSAGE_TYPES", set([param]))
        else:
            self.__class__._MESSAGE_TYPES.add(param)

    def _register(self, param, getter, setter, unit=None, limiter=None):
        if getter:
            self._register_getter(param, getter)

        if setter:
            self._register_setter(param, setter)

        if unit:
            self._units[param] = unit

        if limiter:
            if not limiter(self.get(param)):
                s = "Unable to register limiter. " +\
                    "Value {0} for `{1}` is already out of range"
                raise RuntimeError(s.format(self.get(param), param))
            self._limiters[param] = limiter

        self._register_message(param)

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
