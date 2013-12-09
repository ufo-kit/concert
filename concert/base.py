# -*- coding: utf-8 -*-
"""Core module Parameters"""
import logging
import six
from concert.helpers import memoize
from concert.async import dispatcher, async, wait
from concert.fsm import transition


LOG = logging.getLogger(__name__)


def identity(x):
    return x


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


class Parameter(object):

    """A parameter with getter and setter."""

    def __init__(self, fget=None, fset=None, data=None, source=None,
                 target=None, immediate=None):
        self.name = None
        self.fget = fget
        self.fset = fset
        self.data_args = (data,) if data is not None else ()

        if source or target or immediate:
            self.transition = transition(source, target, immediate)
        else:
            self.transition = None

        self.decorated = None

    @memoize
    def setter_name(self):
        if self.fset:
            return self.fset.__name__

        return '_set_' + self.name

    @memoize
    def getter_name(self):
        if self.fget:
            return self.fget.__name__

        return '_get_' + self.name

    def __get__(self, instance, owner):
        # If we would just call self.fset(value) we would call the method
        # defined in the base class. This is a hack (?) to call the function on
        # the instance where we actually want the function to be called.

        try:
            if self.fget:
                value = self.fget(instance, *self.data_args)
            else:
                value = getattr(instance, self.getter_name())(*self.data_args)

            return value
        except NotImplementedError:
            raise ReadAccessError(self.name)

    def __set__(self, instance, value):
        def log_access(what):
            """Log access."""
            msg = "{}: {}='{}'"
            name = instance.__class__.__name__
            LOG.info(msg.format(name, what, value))

        log_access('try')

        # The same idea as sketched in __get__
        if self.fset:
            self.fset(instance, value, *self.data_args)
        else:
            try:
                setter = getattr(instance, self.setter_name())

                if self.transition and not self.decorated and not hasattr(setter, '_concert_fsm'):
                    self.decorated = self.transition(setter.__func__)

                if self.decorated:
                    self.decorated(instance, value, *self.data_args)
                else:
                    setter(value, *self.data_args)

            except NotImplementedError:
                raise WriteAccessError(self.name)

        log_access('set')


class Quantity(Parameter):

    """A parameter which models a physical quantity."""

    def __init__(self, fget=None, fset=None, unit=None, lower=None, upper=None,
                 conversion=identity, data=None, in_hard_limit=None,
                 source=None, target=None, immediate=None):
        super(Quantity, self).__init__(fget=fget, fset=fset, data=data,
                                       source=source, target=target, immediate=immediate)
        self.unit = unit
        self.default_conversion = conversion
        self.in_hard_limit = in_hard_limit

        self.upper = upper if upper is not None else float('Inf')
        self.lower = lower if lower is not None else -float('Inf')

        if unit and upper is None:
            self.upper = self.upper * unit

        if unit and lower is None:
            self.lower = self.lower * unit

    def is_compatible(self, value):
        try:
            _ = self.unit + value
            return True
        except ValueError:
            return False

    @memoize
    def from_scale(self, instance):
        conversion = self.get_conversion(instance)
        scale = conversion(1)

        if hasattr(scale, 'magnitude'):
            return 1 / (scale / scale.magnitude)

        return 1 / scale

    def get_conversion(self, instance):
        return instance[self.name].conversion

    def convert_to(self, instance, value):
        conversion = self.get_conversion(instance)
        return conversion(value)

    def convert_from(self, instance, value):
        return value * self.from_scale(instance)

    def __get__(self, instance, owner):
        # If we would just call self.fset(value) we would call the method
        # defined in the base class. This is a hack (?) to call the function on
        # the instance where we actually want the function to be called.
        value = super(Quantity, self).__get__(instance, owner)

        return self.convert_from(instance, value)

    def __set__(self, instance, value):
        if self.unit and not self.is_compatible(value):
            msg = "{} of {} can only receive values of unit {} but got {}"
            raise UnitError(
                msg.format(self.name, type(instance), self.unit, value))

        lower = instance[self.name].lower
        upper = instance[self.name].upper

        if not lower <= value <= upper:
            msg = "{} is out of range [{}, {}]"
            raise SoftLimitError(msg.format(value, lower, upper))

        converted = self.convert_to(instance, value)
        super(Quantity, self).__set__(instance, converted)

        if self.in_hard_limit:
            limit_checker = getattr(instance, self.in_hard_limit.__name__)

            if limit_checker():
                msg = "`{}' reached hard limit for {}"
                raise HardLimitError(msg.format(self.name, value))


class ParameterValue(object):

    def __init__(self, instance, parameter):
        self.lock = None
        self._instance = instance
        self._parameter = parameter
        self._saved = []

    def __enter__(self):
        if self.lock is not None:
            self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lock is not None:
            self.lock.release()

    def __lt__(self, other):
        return self._parameter.name < other._parameter.name

    @property
    def name(self):
        return self._parameter.name

    @property
    def data(self):
        return self._parameter.data

    @async
    def get(self):
        return getattr(self._instance, self.name)

    @async
    def set(self, value):
        setattr(self._instance, self.name, value)

    @async
    def stash(self):
        """Save the current value internally on a growing stack.

        If the parameter is writable the current value is saved on a stack and
        to be later retrieved with :meth:`Parameter.restore`.
        """
        self._saved.append(self.get().result())

    def restore(self):
        """Restore the last value saved with :meth:`Parameter.stash`.

        If the parameter can only be read or no value has been saved, this
        operation does nothing.
        """
        if self._saved:
            val = self._saved.pop()
            return self.set(val)


class QuantityValue(ParameterValue):

    def __init__(self, instance, quantity):
        super(QuantityValue, self).__init__(instance, quantity)
        self.lower = quantity.lower
        self.upper = quantity.upper
        self.conversion = quantity.default_conversion

    @property
    def unit(self):
        return self._parameter.unit


class MetaParameterizable(type):

    def __init__(cls, name, bases, dic):
        super(MetaParameterizable, cls).__init__(name, bases, dic)

        def get_base_parameter_names():
            for base in bases:
                if hasattr(base, '_parameter_names'):
                    return getattr(base, '_parameter_names')
            return {}

        if not hasattr(cls, '_parameter_names'):
            setattr(cls, '_parameter_names', get_base_parameter_names())

        for attr_name, attr_type in dic.items():
            if isinstance(attr_type, Parameter):
                attr_type.name = attr_name
                cls._parameter_names[(cls, attr_name)] = attr_type


class Parameterizable(six.with_metaclass(MetaParameterizable, object)):

    """
    Collection of parameters.

    For each class of type :class:`Parameterizable`, :class:`Parameter` can be
    set as class attributes ::

        class Device(Parameterizable):

            def get_something(self):
                return 'something'

            something = Parameter(get_something)

    There is a simple :class:`Parameter` and a parameter which models a
    physical quantity :class:`Quantity`.

    A :class:`Parameterizable` is iterable and returns its parameters of type
    :class:`ParameterValue` or its subclasses ::

        for param in device:
            print("name={}".format(param.name))

    To access a single name parameter object, you can use the ``[]`` operator::

        param = device['position']
        print param.is_readable()

    If the parameter name does not exist, a :class:`.ParameterError` is raised.

    Each parameter value is accessible as a property. If a device has a
    position it can be read and written with::

        param.position = 0 * q.mm
        print param.position
    """

    def __init__(self):
        if not hasattr(self, '_params'):
            self._params = {}

        for (cls, name), parameter in self._parameter_names.items():
            if not isinstance(self, cls):
                continue

            self._install_parameter(name, parameter)

    def __str__(self):
        from concert.session.utils import get_default_table

        table = get_default_table(["Parameter", "Value"])
        table.border = False

        for param in self:
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

    def install_parameters(self, params):
        """Install parameters at run-time.

        *params* is a dictionary mapping parameter names to :class:`.Parameter`
        objects.
        """
        for name, param in params.items():
            param.name = name
            self._install_parameter(name, param)

            # Install param as a property, so that it can be accessed via
            # object-dot notation.
            setattr(self.__class__, name, param)

    def _install_parameter(self, name, param):
        if param.__class__ == Quantity:
            value = QuantityValue(self, param)
        elif param.__class__ == Parameter:
            value = ParameterValue(self, param)

        self._params[name] = value

        def setter_not_implemented(value, *args):
            raise NotImplementedError

        def getter_not_implemented(*args):
            raise NotImplementedError

        setattr(self, 'set_' + name, value.set)
        setattr(self, 'get_' + name, value.get)

        if not hasattr(self, '_set_' + name):
            setattr(self, '_set_' + name, setter_not_implemented)

        if not hasattr(self, '_get_' + name):
            setattr(self, '_get_' + name, getter_not_implemented)

    @async
    def stash(self):
        """
        Save all writable parameters that can be restored with
        :meth:`Parameterizable.restore`.

        The values are stored on a stacked, hence subsequent saved states can
        be restored one by one.
        """
        wait((param.stash() for param in self))

    @async
    def restore(self):
        """Restore all parameters saved with :meth:`Parameterizable.stash`."""
        wait((param.restore() for param in self))


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
