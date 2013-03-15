"""
The mother of all bases. The lowest level object definition and functionality.
"""
import threading
from concert.events.dispatcher import dispatcher
from threading import Event
from logbook import Logger


log = Logger(__name__)


class ConcertObject(object):
    """
    Base class for where a set of parameters can be set. Events are produced
    when a setter methods is called. The class provides functionality for
    listening to messages.

    A :class:`Device` consists of optional getters, setters, limiters and units
    for named parameters. Implementations first call :meth:`__init__` and then
    :meth:`_register` to add or supplement a parameter.
    """
    def __init__(self):
        self._setters = {}
        self._getters = {}
        self._limiters = {}
        self._units = {}

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
            s = "`{0}' can only receive values of unit {1}"
            raise ValueError(s.format(param, self._units[param]))

        if not self._value_is_in_range(param, value):
            s = "Value {0} for `{1}` is out of range"
            raise ValueError(s.format(value, param))

        msg = "{0}: set {1}='{2}' blocking='{3}'"
        log.info(msg.format(str(self), param, value, blocking))

        setter = self._setters[param]
        return self._launch(param, setter, (value,), blocking)

    def _unit_is_compatible(self, param, value):
        if not param in self._units:
            return True

        try:
            self._units[param] + value
            return True
        except:
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

        # Register a message on which one can wait.
        message_name = param.upper()
        if not hasattr(self.__class__, message_name):
            setattr(self.__class__, message_name, param)
            if not hasattr(self.__class__, "_MESSAGE_TYPES"):
                setattr(self.__class__, "_MESSAGE_TYPES", set([message_name]))
            else:
                self.__class__._MESSAGE_TYPES.add(message_name)

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
        dispatcher.subscribe([(self, message)], callback)

    def unsubscribe(self, message, callback):
        """
        Unsubscribe from a *message*.

        *callback* is a function which is unsubscribed from a particular
        *message* coming from this object.
        """
        dispatcher.unsubscribe([(self, message)], callback)

    def _launch(self, param, action, args=(), blocking=False):
        """Launch *action* with *args* with message and event handling after
        the action finishes.

        If *blocking* is ``True``, *action* will be called like an ordinary
        function otherwise a thread will be started. *args* must be a tuple of
        arguments that is then unpacked and passed to *action* at _launch time.
        The *action* itself must be blocking.
        """
        def _action(event, args):
            """Call action and handle its finish."""
            action(*args)
            event.set()
            self.send(getattr(self.__class__, param.upper()))

        event = Event()
        launch(_action, (event, args), blocking)
        return event


def launch(action, args=(), blocking=False):
    """Launch *action* with *args*.

    If *blocking* is ``True``, *action* will be called like an ordinary
    function otherwise a thread will be started. *args* must be a tuple of
    arguments that is then unpacked and passed to *action* at launch time.
    The *action* itself must be blocking.
    """
    if blocking:
        action(*args)
    else:
        thread = threading.Thread(target=action, args=args)
        thread.daemon = True
        thread.start()
