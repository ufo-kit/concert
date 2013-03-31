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
from concurrent.futures import ThreadPoolExecutor, wait
from concert.events.dispatcher import dispatcher
from threading import Event
from logbook import Logger


executor = ThreadPoolExecutor(max_workers=2)

log = Logger(__name__)


class UnitError(ValueError):
    """Raised when an operation is passed value with an incompatible unit"""
    pass


class LimitError(Exception):
    """Raised when an operation is passed a value that exceeds a limit"""
    pass


class ParameterError(Exception):
    """Raised when a parameter is accessed that does not exists"""
    def __init__(self, parameter):
        msg = "`{0}' is not a parameter".format(parameter)
        super(ParameterError, self).__init__(msg)


class ReadAccessError(Exception):
    """Raised when user tries to change a parameter that cannot be written"""
    def __init__(self, parameter):
        log.warn("Invalid read access on {0}".format(parameter))
        msg = "parameter `{0}' cannot be read".format(parameter)
        super(ReadAccessError, self).__init__(msg)


class WriteAccessError(Exception):
    """Raised when user tries to read a parameter that cannot be read"""
    def __init__(self, parameter):
        log.warn("Invalid write access on {0}".format(parameter))
        msg = "parameter `{0}' cannot be written".format(parameter)
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

    CHANGED = 'changed'

    def __init__(self, name, fget=None, fset=None,
                 unit=None, limiter=None,
                 doc=None, owner_only=False):
        self.name = name
        self.unit = unit
        self.limiter = limiter
        self.owner = None
        self._owner_only = owner_only
        self._fset = fset
        self._fget = fget
        self.__doc__ = doc

    def get(self):
        """Try to read and return the current value.

        If the parameter cannot be read, :py:class:`ReadAccessError` is raised.
        """
        if not self.is_readable():
            raise ReadAccessError(self.name)

        return executor.submit(self._fget)

    def set(self, value, owner=None):
        """Try to write *value*.

        If the parameter cannot be written, :py:class:`WriteAccessError` is
        raised. Once the value has been written on the device, all associated
        callbacks are called and a message is placed on the dispatcher bus.
        """
        if self._owner_only and owner != self.owner:
            raise WriteAccessError(self.name)

        if not self.is_writable():
            raise WriteAccessError(self.name)

        if self.unit and not value_compatible(value, self.unit):
            msg = "`{0}' can only receive values of unit {1} but got {2}"
            raise UnitError(msg.format(self.name, self.unit, value))

        if self.limiter and not self.limiter(value):
            msg = "{0} for `{1}' is out of range"
            raise LimitError(msg.format(value, self.name))

        def log_access(what):
            msg = "{0}: {1} {2}='{3}'"
            name = self.owner.__class__.__name__
            log.info(msg.format(name, what, self.name, value))

        def finish_setting(future):
            log_access('set')
            self.notify()

        log_access('try')
        future = executor.submit(self._fset, value)
        future.add_done_callback(finish_setting)
        return future

    def notify(self):
        """Notify that the parameter value has changed."""
        dispatcher.send(self, self.CHANGED)

    def is_readable(self):
        """Return `True` if parameter can be read."""
        return self._fget is not None

    def is_writable(self):
        """Return `True` if parameter can be written."""
        return self._fset is not None


class _ProppedParameter(object):
    def __init__(self, parameter):
        self.parameter = parameter

    def __get__(self, instance, owner):
        future = self.parameter.get()
        return future.result()

    def __set__(self, instance, value):
        future = self.parameter.set(value)
        wait([future])


class Device(object):
    """
    The :class:`Device` provides synchronous access to a real hardware device,
    such as a motor, a pump or a camera.

    A :class:`Device` is iterable and returns its parameters of type
    :class:`Parameter` ::

        for param in device:
            msg = "{0} readable={1}, writable={2}"
            print(msg.format(param.name,
                             param.is_readable(), param.is_writable())

    To access a single name parameter object, you can use the ``[]`` operator::

        param = device['position']
        print param.is_readable()

    Each parameter value is accessible as a property. If a device has a
    position it can be read and written with::

        param.position = 0 * q.mm
        print param.position
    """
    def __init__(self, parameters=None):
        self._params = {}

        if parameters:
            for parameter in parameters:
                parameter.owner = self
                self.add_parameter(parameter)

        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, type, value, traceback):
        self._lock.release()

    def __str__(self):
        params = [(param.name, param.get().result())
                  for param in self
                  if param.is_readable()]
        params.sort()
        return '\n'.join("%s = %s" % p for p in params)

    def __iter__(self):
        for param in self._params.values():
            yield param

    def __getitem__(self, param):
        if param not in self._params:
            raise ParameterError(param)

        return self._params[param]

    def add_parameter(self, parameter):
        """Add *parameter* to device and install a property of the same name
        for reading and/or writing it."""
        self._params[parameter.name] = parameter
        parameter.owner = self
        setattr(self.__class__, parameter.name, _ProppedParameter(parameter))


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
    def __init__(self):
        pass

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


# def wait(events, timeout=None):
#     """Wait until all *events* finished."""
#     for event in events:
#         event.wait(timeout)
