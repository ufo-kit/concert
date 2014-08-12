# -*- coding: utf-8 -*-
"""Core module Parameters"""
import numpy as np
import logging
import six
import functools
import inspect
import types
import threading
from concert.helpers import memoize, busy_wait
from concert.async import async, wait
from concert.quantities import q


LOG = logging.getLogger(__name__)


def identity(x):
    return x


def _setter_not_implemented(value, *args):
    raise AccessorNotImplementedError


def _is_compatible(unit, value):
    try:
        unit + value
        return True
    except ValueError:
        return False


def _getter_not_implemented(*args):
    raise AccessorNotImplementedError


def _execute_func(func, instance, *args, **kwargs):
    """Execute *func* irrespective of whether it is a function or a method. *instance*
    is discarded if *func* is a function, otherwise it is used as a first real argument.
    """
    if isinstance(func, types.MethodType):
        result = func(*args, **kwargs)
    else:
        result = func(instance, *args, **kwargs)

    return result


class FSMError(Exception):
    """All errors connected with the finite state machine"""


class TransitionNotAllowed(FSMError):
    pass


class StateError(Exception):

    """Raised in state check functions of devices."""

    def __init__(self, error_state, msg=None):
        self.state = error_state
        self.msg = msg if msg else "Got into `{}'".format(error_state)

    def __str__(self):
        return self.msg


class UnitError(ValueError):

    """Raised when an operation is passed value with an incompatible unit."""
    pass


class LimitError(ValueError):

    """Raised when an operation is passed a value that exceeds a limit."""
    pass


class SoftLimitError(LimitError):

    """Raised when a soft limit is hit on the device."""
    pass


class HardLimitError(StateError):

    """Raised when device goes into hardlimit error state."""

    def __init__(self, error_state='hard-limit', msg=None):
        super(HardLimitError, self).__init__(error_state, msg)


class ParameterError(Exception):

    """Raised when a parameter is accessed that does not exists."""

    def __init__(self, parameter):
        msg = "`{0}' is not a parameter".format(parameter)
        super(ParameterError, self).__init__(msg)


class AccessorNotImplementedError(NotImplementedError):

    """Raised when a setter or getter is not implemented."""


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


class LockError(Exception):

    """Raised when parameter is locked."""

    pass


def transition(immediate=None, target=None):
    """Change software state of a device to *immediate*. After the function
    execution finishes change the state to *target*.
    """
    def wrapped(func):
        @functools.wraps(func)
        def call_func(instance, *args, **kwargs):
            if not hasattr(instance, 'state'):
                raise FSMError('Changing state requires state parameter')

            # Store the original in case target is None
            target_state = target if target else instance.state

            if immediate:
                setattr(instance, '_state_value', immediate)

            try:
                result = _execute_func(func, instance, *args, **kwargs)
                setattr(instance, '_state_value', target_state)
            except StateError as error:
                setattr(instance, '_state_value', error.state)
                raise error

            return result

        return call_func

    return wrapped


def check(source='*', target=None):
    """
    Decorates a method for checking the device state.

    *source* denotes the source state that must be present at the time of
    invoking the decorated method. *target* is the state that the state object
    will be after successful completion of the method or a list of possible
    target states.
    """
    def wrapped(func):
        sources = [source] if isinstance(source, str) else source
        targets = [target] if isinstance(target, str) else target

        @functools.wraps(func)
        def call_func(instance, *args, **kwargs):
            if not hasattr(instance, 'state'):
                raise FSMError('Transitioning requires state parameter')

            if instance.state not in sources and '*' not in sources:
                msg = "Current state `{}' not in `{}'".format(instance.state, sources)
                raise TransitionNotAllowed(msg)

            result = _execute_func(func, instance, *args, **kwargs)

            # Check if the device got into an allowed state after the check
            final = instance.state
            if final not in targets:
                msg = "Final state `{}' not in `{}'".format(final, targets)
                raise TransitionNotAllowed(msg)
            return result

        return call_func

    return wrapped


class Parameter(object):

    """A parameter with getter and setter.

    Parameters are similar to normal Python properties and can additionally
    trigger state checks. If *fget* or *fset* is not given, you must
    implement the accessor functions named `_set_name` and `_get_name`::

        from concert.base import Parameter, State

        class SomeClass(object):

            state = State(default='standby')

            def actual(self):
                return 'moving'

            param = Parameter(check=check(source='standby',
                                                    target=['standby', 'moving'],
                                                    check=actual))

            def _set_param(self, value):
                pass

            def _get_param(self):
                pass

    When a :class:`.Parameter` is attached to a class, you can modify it by
    accessing its associated :class:`.ParameterValue` with a dictionary
    access::

        obj = SomeClass()
        print(obj['param'])
    """

    def __init__(self, fget=None, fset=None, data=None, check=None, help=None):
        """
        *fget* is a callable that is called when reading the parameter. *fset*
        is called when the parameter is written to.

        *data* is passed to the state check function.

        *check* is a :func:`.check` that changes states when a value
        is written to the parameter.

        *help* is a string describing the parameter in more detail.
        """
        self.name = None
        self.fget = fget
        self.fset = fset
        self.data_args = (data,) if data is not None else ()
        self.check = check
        self.decorated = None
        self.help = help

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

    def __repr__(self):
        return str(self.help)

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
        except AccessorNotImplementedError:
            raise ReadAccessError(self.name)

    def __set__(self, instance, value):
        def log_access(what):
            """Log access."""
            msg = "{}: {}: {}='{}'"
            name = instance.__class__.__name__
            LOG.info(msg.format(name, what, self.name, value))

        log_access('try')

        if instance[self.name].locked:
            raise LockError("Parameter `{}' is locked for writing".format(self))

        if self.fset:
            self.fset(instance, value, *self.data_args)
        else:
            func = getattr(instance, '_set_' + self.name)

            if self.check and not hasattr(func, '_is_checked'):
                func = self.check(func)
                func._is_checked = True
                setattr(instance, '_set_' + self.name, func)

            try:
                if isinstance(func, types.MethodType):
                    func(value, *self.data_args)
                else:
                    func(instance, value, *self.data_args)
            except AccessorNotImplementedError:
                raise WriteAccessError(self.name)

        log_access('set')


class State(Parameter):

    """
    Finite state machine.

    Use this on a class, to keep some sort of known state. In order to enforce
    restrictions, you would decorate methods on the class with
    :func:`.check`::

        class SomeObject(object):

            state = State(default='standby')

            @check(source='*', target='moving')
            def move(self):
                pass

    In case your device doesn't provide information on its state you can use
    the :func:`.transition` to store the state in an instance of your device::

        @transition(immediate='moving', target='standby')
        def _set_some_param(self, param_value):
            # when the method starts device state is set to *immediate*
            # long operation goes here
            pass
            # the state is set to *target* in the end

    Accessing the state variable will return the current state value, i.e.::

        obj = SomeObject()
        assert obj.state == 'standby'

    The state cannot be set explicitly by::

        obj.state = 'some_state'

    but the object needs to provide methods which transition out of
    states, the same holds for transitioning out of error states.
    If the :meth:`_get_state` method is implemented in the device
    it is always used to get the state, otherwise the state is stored
    in software.
    """

    def __init__(self, default=None, fget=None, fset=None, data=None, check=None, help=None):
        super(State, self).__init__(fget=fget, help=help)
        self.default = default

    def __get__(self, instance, owner):
        try:
            return super(State, self).__get__(instance, owner)
        except ReadAccessError:
            if self.default is None:
                raise FSMError('Software state must have a default value')
            return self._value(instance)

    def __set__(self, instance, value):
        raise AttributeError('State cannot be set')

    def _value(self, instance):
        if not hasattr(instance, '_state_value'):
            setattr(instance, '_state_value', self.default)

        return getattr(instance, '_state_value')


class Quantity(Parameter):

    """A :class:`.Parameter` associated with a unit."""

    def __init__(self, unit, fget=None, fset=None, lower=None, upper=None,
                 data=None, check=None, help=None):
        """
        *fget*, *fset*, *data*, *check* and *help* are identical to the
        :class:`.Parameter` constructor arguments.

        *unit* is a Pint quantity. *lower* and *upper* denote soft limits
        between the :class:`.Quantity` values can lie.
        """
        super(Quantity, self).__init__(fget=fget, fset=fset, data=data,
                                       check=check, help=help)
        self.unit = unit

        self.upper = upper if upper is not None else float('Inf')
        self.lower = lower if lower is not None else -float('Inf')

        if upper is None:
            self.upper = self.upper * unit

        if lower is None:
            self.lower = self.lower * unit

    def convert(self, value):
        return value.to(self.unit)

    def __get__(self, instance, owner):
        # If we would just call self.fset(value) we would call the method
        # defined in the base class. This is a hack (?) to call the function on
        # the instance where we actually want the function to be called.
        value = super(Quantity, self).__get__(instance, owner)

        return self.convert(value)

    def __set__(self, instance, value):
        if not _is_compatible(self.unit, value):
            msg = "{} of {} can only receive values of unit {} but got {}"
            raise UnitError(
                msg.format(self.name, type(instance), self.unit, value))

        lower = instance[self.name].lower
        upper = instance[self.name].upper

        def leq(a, b):
            """check a <= b"""
            if not hasattr(a, 'shape') or len(a.shape) == 0:
                return a <= b
            else:
                # Vector data type
                valid_a = np.invert(np.isnan(a.magnitude))
                valid_b = np.invert(np.isnan(b.magnitude))
                valid = valid_a & valid_b
                test_a = a[valid]
                test_b = b[valid]
                return np.all(test_a.to_base_units().magnitude <= test_b.to_base_units().magnitude)

        if not leq(lower, value) or not leq(value, upper):
            msg = "{} is out of range [{}, {}]"
            raise SoftLimitError(msg.format(value, lower, upper))

        converted = self.convert(value)
        super(Quantity, self).__set__(instance, converted)


def quantity(unit=None, lower=None, upper=None, data=None, check=None, help=None):
    """
    Decorator for read-only quantity functions.

    Device authors can add additional read-only quantities to a specific device
    by applying this decorator to a function::

        class SomeDevice(Device):
            @quantity(unit=1*q.mm, lower=0*q.mm, upper=2*q.mm)
            def position(self):
                pass

    The arguments correspond to those of :class:`.Quantity`.
    """

    def wrapper(func):
        # Get help info from doc string if no explicit help was passed.
        doc = help if help else inspect.getdoc(func)

        return Quantity(fget=func, unit=unit, lower=lower, upper=upper,
                        data=data, check=check, help=doc)

    return wrapper


class ParameterValue(object):

    """Value object of a :class:`.Parameter`."""

    def __init__(self, instance, parameter):
        self._lock = threading.Lock()
        self._locked = False
        self._instance = instance
        self._parameter = parameter
        self._saved = []

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()

    def __lt__(self, other):
        return self._parameter.name < other._parameter.name

    def __repr__(self):
        return self.info_table.get_string()

    @property
    def info_table(self):
        from concert.session.utils import get_default_table
        locked = "yes" if self.locked else "no"
        table = get_default_table(["attribute", "value"])
        table.header = False
        table.border = False
        table.add_row(["info", self._parameter.help])
        table.add_row(["locked", locked])
        return table

    @property
    def name(self):
        return self._parameter.name

    @property
    def data(self):
        return self._parameter.data

    @property
    def writable(self):
        """Return True if the parameter is writable."""
        return getattr(self._instance, '_set_' + self.name) is not _setter_not_implemented

    @async
    def get(self, wait_on=None):
        """
        Get concrete *value* of this object.

        If *wait_on* is not None, it must be a future on which this method
        joins.
        """
        if wait_on:
            wait_on.join()

        return getattr(self._instance, self.name)

    @async
    def set(self, value, wait_on=None):
        """
        Set concrete *value* on the object.

        If *wait_on* is not None, it must be a future on which this method
        joins.
        """
        if wait_on:
            wait_on.join()

        setattr(self._instance, self.name, value)

    @async
    def stash(self):
        """Save the current value internally on a growing stack.

        If the parameter is writable the current value is saved on a stack and
        to be later retrieved with :meth:`.ParameterValue.restore`.
        """
        if not self.writable:
            raise ParameterError("Parameter `{}' is not writable".format(self.name))

        self._saved.append(self.get().result())

    def restore(self):
        """Restore the last value saved with :meth:`.ParameterValue.stash`.

        If the parameter can only be read or no value has been saved, this
        operation does nothing.
        """
        if not self.writable:
            raise ParameterError("Parameter `{}' is not writable".format(self.name))

        if self._saved:
            val = self._saved.pop()
            return self.set(val)

    @property
    def locked(self):
        """Return True if the parameter is locked for writing."""
        return self._locked

    def lock(self, permanent=False):
        """Lock parameter for writing. If *permament* is True the parameter
        cannot be unlocked anymore.
        """
        def unlock_not_allowed():
            raise LockError("Parameter `{}' cannot be unlocked".format(self))

        self._locked = True
        if permanent:
            self.unlock = unlock_not_allowed

    def unlock(self):
        """Unlock parameter for writing."""
        self._locked = False

    def wait(self, value, sleep_time=1e-1 * q.s, timeout=None):
        """Wait until the parameter value is *value*. *sleep_time* is the time to sleep
        between consecutive checks. *timeout* specifies the maximum waiting time.
        """
        condition = lambda: self.get().result() == value
        busy_wait(condition, sleep_time=sleep_time, timeout=timeout)


class QuantityValue(ParameterValue):

    def __init__(self, instance, quantity):
        super(QuantityValue, self).__init__(instance, quantity)
        self._lower = quantity.lower
        self._upper = quantity.upper
        self._limits_locked = False

    def lock_limits(self, permanent=False):
        """Lock limits, if *permanent* is True the limits cannot be unlocker anymore."""
        def unlock_not_allowed():
            raise LockError('Limits are locked permanently')

        self._limits_locked = True
        if permanent:
            self.unlock_limits = unlock_not_allowed

    def unlock_limits(self):
        """Unlock limits."""
        self._limits_locked = False

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        if self._limits_locked:
            raise LockError('lower limit locked')
        else:
            self._lower = value

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, value):
        if self._limits_locked:
            raise LockError('upper limit locked')
        else:
            self._upper = value

    @property
    def info_table(self):
        table = super(QuantityValue, self).info_table
        table.add_row(["lower", self.lower])
        table.add_row(["upper", self.upper])
        return table

    @property
    def unit(self):
        return self._parameter.unit

    def wait(self, value, eps=None, sleep_time=1e-1 * q.s, timeout=None):
        """Wait until the parameter value is *value*. *eps* is the allowed discrepancy between the
        actual value and *value*. *sleep_time* is the time to sleep between consecutive checks.
        *timeout* specifies the maximum waiting time.
        """
        def eps_condition():
            """Take care of rountind errors"""
            diff = np.abs((self.get().result() - value).to(eps.units))
            return diff < eps

        condition = lambda: self.get().result() == value if eps is None else eps_condition
        busy_wait(condition, sleep_time=sleep_time, timeout=timeout)


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

    For each class of type :class:`.Parameterizable`, :class:`.Parameter` can
    be set as class attributes ::

        class Device(Parameterizable):

            def get_something(self):
                return 'something'

            something = Parameter(get_something)

    There is a simple :class:`.Parameter` and a parameter which models a
    physical quantity :class:`.Quantity`.

    A :class:`.Parameterizable` is iterable and returns its parameters of type
    :class:`.ParameterValue` or its subclasses ::

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
        if isinstance(param, Quantity):
            value = QuantityValue(self, param)
        elif isinstance(param, Parameter):
            value = ParameterValue(self, param)

        self._params[name] = value

        setattr(self, 'set_' + name, value.set)
        setattr(self, 'get_' + name, value.get)

        if not hasattr(self, '_set_' + name):
            setattr(self, '_set_' + name, _setter_not_implemented)

        if not hasattr(self, '_get_' + name):
            setattr(self, '_get_' + name, _getter_not_implemented)

    @async
    def stash(self):
        """
        Save all writable parameters that can be restored with
        :meth:`.Parameterizable.restore`.

        The values are stored on a stacked, hence subsequent saved states can
        be restored one by one.
        """
        wait((param.stash() for param in self if param.writable))

    @async
    def restore(self):
        """Restore all parameters saved with :meth:`.Parameterizable.stash`."""
        wait((param.restore() for param in self if param.writable))

    def lock(self, permanent=False):
        """Lock all the parameters for writing. If *permanent* is True, the
        parameters cannot be unlocked anymore.
        """
        for param in self:
            param.lock(permanent=permanent)

    def unlock(self):
        """Unlock all the parameters for writing."""
        for param in self:
            param.unlock()


