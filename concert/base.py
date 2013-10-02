# -*- coding: utf-8 -*-
"""
A *device* is a software abstraction for a piece of hardware that can be
controlled.

Each device consists of a set of named :class:`Parameter` instances and
device-specific methods. If you know the parameter name, you can get
a reference to the parameter object by using the index operator::

    pos_parameter = motor['position']

To set and get parameters explicitly , you can use the :meth:`Parameter.get`
and :meth:`Parameter.set` methods::

    pos_parameter.set(1 * q.mm)
    print (pos_parameter.get().result())

Both methods will return a *Future*. A future is a promise that a result will
be delivered when asked for. In the mean time other things can and should
happen concurrently. As you can see, to get the result of a future you call its
``result()`` method.

An easier way to set and get parameter values are properties via the
dot-name-notation::

    motor.position = 1 * q.mm
    print (motor.position)

As you can see, accessing parameters this way will *always be synchronous* and
*block* execution until the value is set or fetched.

Parameter objects are not only used to communicate with a device but also carry
meta data information about the parameter. The most important ones are
:attr:`Parameter.name`, :attr:`Parameter.unit` and
:attr:`Parameter.in_hard_limit` as well as the doc string describing the
parameter. Moreover, parameters can be queried for access rights using
:meth:`Parameter.is_readable` and :meth:`Parameter.is_writable`.

To get all parameters of an object, you can iterate over the device itself ::

    for param in motor:
        print("{0} => {1}".format(param.unit, param.name))
"""
import re
from logbook import Logger
from functools import wraps
from concert.helpers import dispatcher, async
from concert.ui import get_default_table


LOG = Logger(__name__)


class UnitError(ValueError):

    """Raised when an operation is passed value with an incompatible unit."""
    pass


class LimitError(ValueError):

    """Raised when an operation is passed a value that exceeds a limit."""
    pass


class SoftLimitError(LimitError):

    """Raised when a soft limit is hit on the device."""
    pass


class HardLimitError(LimitError):

    """Raised when a hard limit is hit on the device."""
    pass


class ParameterError(Exception):

    """Raised when a parameter is accessed that does not exists."""

    def __init__(self, parameter):
        msg = "`{0}' is not a parameter".format(parameter)
        super(ParameterError, self).__init__(msg)


class ReadAccessError(Exception):

    """Raised when user tries to change a parameter that cannot be written."""

    def __init__(self, parameter):
        LOG.warn("Invalid read access on {0}".format(parameter))
        msg = "parameter `{0}' cannot be read".format(parameter)
        super(ReadAccessError, self).__init__(msg)


class WriteAccessError(Exception):

    """Raised when user tries to read a parameter that cannot be read."""

    def __init__(self, parameter):
        LOG.warn("Invalid write access on {0}".format(parameter))
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
        _ = unit + value
        return True
    except ValueError:
        return False


def parameter_name_valid(name):
    """Check if a parameter *name* is correct and return ``True`` if so."""
    expr = r'^[a-zA-Z]+[a-zA-Z0-9_-]*$'
    return re.match(expr, name) is not None


class Parameter(object):

    """
    A parameter with a *name* and an optional *unit* and *in_hard_limit*.

    .. py:attribute:: name

        The name of the parameter.

    .. py:attribute:: unit

        The unit that is expected when setting a value and that is returned. If
        a unit is not compatible, a :class:`.UnitError` will be raised.

    .. py:attribute:: in_hard_limit

        A callable that receives the value and returns True or False, depending
        if the value is out of limits or not.

    .. py:attribute:: upper

        Upper soft limit that is checked before setting the value.

    .. py:attribute:: lower

        Lower soft limit that is checked before setting the value.
    """

    CHANGED = 'changed'

    def __init__(self, name, fget=None, fset=None,
                 unit=None, in_hard_limit=None,
                 upper=None, lower=None,
                 doc=None):

        if not parameter_name_valid(name):
            raise ValueError('{0} is not a valid parameter name'.format(name))

        self.name = name
        self.unit = unit
        self.in_hard_limit = in_hard_limit
        self.owner = None
        self.lock = None
        self._fset = fset
        self._fget = fget
        self._value = None
        self.__doc__ = doc

        self.upper = upper if upper is not None else float('Inf')
        self.lower = lower if lower is not None else -float('Inf')

        if unit and upper is None:
            self.upper = self.upper * unit

        if unit and lower is None:
            self.lower = self.lower * unit

    def __enter__(self):
        if self.lock is not None:
            self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lock is not None:
            self.lock.release()

    def __lt__(self, other):
        return str(self) <= str(other)

    def update_unit(self, unit):
        self.unit = unit
        self.upper = self.upper.magnitude * unit
        self.lower = self.lower.magnitude * unit

    @async
    def get(self):
        """
        get()
        Read and return the current value.

        If the parameter cannot be read, :class:`.ReadAccessError` is raised.
        """
        if not self.is_readable():
            raise ReadAccessError(self.name)

        if self._is_simple():
            return self._value

        return self._fget()

    @async
    def set(self, value, owner=None):
        """
        set(value, owner=None)
        Write *value*.

        If the parameter cannot be written, :class:`.WriteAccessError` is
        raised. If :attr:`unit` is set and not compatible with *value*,
        :class:`.UnitError` is raised.

        Once the value has been written on the device, all associated
        callbacks are called and a message is placed on the dispatcher bus.
        """
        if owner is not None and owner != self.owner:
            raise WriteAccessError(self.name)

        if not self.is_writable():
            raise WriteAccessError(self.name)

        if self.unit and not value_compatible(value, self.unit):
            msg = "`{0}' can only receive values of unit {1} but got {2}"
            raise UnitError(msg.format(self.name, self.unit, value))

        if not self.lower <= value <= self.upper:
            msg = "{0} for `{1}' is out of range [{2}, {3}]"
            raise SoftLimitError(msg.format(value, self.name,
                                            self.lower, self.upper))

        def log_access(what):
            """Log access."""
            msg = "{0}: {1} {2}='{3}'"
            name = self.owner.__class__.__name__
            LOG.info(msg.format(name, what, self.name, value))

        log_access('try')

        if self._is_simple():
            self._value = value
        else:
            self._fset(value)

            if self.in_hard_limit and self.in_hard_limit():
                msg = "{0} for `{1}' is out of range"
                raise HardLimitError(msg.format(value, self.name))

        log_access('set')
        self.notify()

    def notify(self):
        """Notify that the parameter value has changed."""
        dispatcher.send(self, self.CHANGED)

    def _is_simple(self):
        return self._fget is None and self._fset is None

    def is_readable(self):
        """Return `True` if parameter can be read."""
        return self._fget is not None or self._is_simple()

    def is_writable(self):
        """Return `True` if parameter can be written."""
        return self._fset is not None or self._is_simple()


class Parameterizable(object):

    """Collection of parameters.

    A :class:`Parameterizable` is iterable and returns its parameters of type
    :class:`Parameter` ::

        for param in device:
            msg = "{0} readable={1}, writable={2}"
            print(msg.format(param.name,
                             param.is_readable(), param.is_writable())

    To access a single name parameter object, you can use the ``[]`` operator::

        param = device['position']
        print param.is_readable()

    If the parameter name does not exist, a :class:`.ParameterError` is raised.

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

    def __str__(self):
        table = get_default_table(["Parameter", "Value"])
        table.border = False
        readable = (param for param in self if param.is_readable())

        for param in readable:
            table.add_row([param.name, str(param.get().result())])

        return table.get_string(sortby="Parameter")

    def __repr__(self):
        return '\n'.join([super(Parameterizable, self).__repr__(),
                         str(self)])

    def __iter__(self):
        for param in sorted(self._params.values()):
            yield param

    def __getitem__(self, param):
        if param not in self._params:
            raise ParameterError(param)

        return self._params[param]

    def add_parameter(self, parameter):
        """Add *parameter*.

        This will install a property :attr:`.Parameter.name` with its getter
        set to :meth:`.Parameter.get` and the setter set to
        :meth:`.Parameter.get`.  The name will only consist of underscores.

        Furthermore, two functions `set_parameter_name` and
        `get_parameter_name` will be installed that are asynchronous.
        """
        self._params[parameter.name] = parameter
        parameter.owner = self
        underscored = parameter.name.replace('-', '_')

        def _getter(self):
            return self[parameter.name].get().result()

        def _setter(self, value):
            self[parameter.name].set(value).wait()

        setattr(self.__class__, underscored, property(_getter, _setter))

        if parameter.is_readable():
            setattr(self, 'get_%s' % underscored, self[parameter.name].get)

        if parameter.is_writable():
            setattr(self, 'set_%s' % underscored, self[parameter.name].set)


class Process(Parameterizable):

    """Base process."""

    def __init__(self, params=None):
        super(Process, self).__init__(params)

    @async
    def run(self):
        """run()

        Run the process. The result depends on the actual process.
        """
        raise NotImplementedError
