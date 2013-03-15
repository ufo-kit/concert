from logbook import Logger
from concert.base import ConcertObject, launch


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


class Device(ConcertObject):
    """Base class for controllable devices for which a set of parameters can be
    set.

    A :class:`Device` consists of optional getters, setters, limiters and units
    for named parameters. Implementations first call :meth:`__init__` and then
    :meth:`_register` to add or supplement a parameter.
    """

    def __init__(self):
        super(Device, self).__init__()
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
        self._register_getter(param, getter)
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
