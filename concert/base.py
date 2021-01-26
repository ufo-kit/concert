# -*- coding: utf-8 -*-
"""Core module Parameters"""
import numpy as np
import logging
import functools
import inspect
import types
import threading
from concert.helpers import hasattr_raise_exceptions, memoize
from concert.async import async, wait, busy_wait
from concert.quantities import q


LOG = logging.getLogger(__name__)


def identity(x):
    return x


def _setter_not_implemented(value, *args):
    raise AccessorNotImplementedError


def _is_compatible(unit, value):
    try:
        1 * unit + value
        return True
    except ValueError:
        return False


def _getter_not_implemented(*args):
    raise AccessorNotImplementedError


def _getter_target_not_implemented(*args):
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


def _find_object_by_name(instance):
    """Find variable name by *instance*. This is supposed to be used only in the
    :meth:`.Parameter.__set__`.
    """
    def _find_in_dict(dictionary):
        for (obj_name, obj) in dictionary.items():
            try:
                if obj == instance:
                    return obj_name
            except:
                # This is not so crucial, just debug level
                LOG.debug('Error trying to find object {} of type {}'.format(obj_name, type(obj)))

    instance_name = None

    frames = inspect.stack()
    try:
        # Skip us and Parameter.__set__
        for i in range(2, len(frames)):
            # First look in the globals
            instance_name = _find_in_dict(frames[i][0].f_globals)
            if not instance_name:
                # If not found, look in the locals (e.g. devices instantiated in funcions)
                instance_name = _find_in_dict(frames[i][0].f_locals)
            # We can't know at which index of the stack is the correct name, e.g. if a device sets a
            # parameter in a constructor we need to bump index by one. Thus, use blacklist names
            # which won't be picked up because they are most probably used by concert or it's
            # extensions internally.
            if instance_name and instance_name not in ['instance', 'self']:
                break
    finally:
        # Cleanup as python docs suggest
        del frames

    return instance_name


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
            if not hasattr_raise_exceptions(instance, 'state'):
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
            if not hasattr_raise_exceptions(instance, 'state'):
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

        from concert.base import Parameter, State, check

        class SomeClass(object):

            state = State(default='standby')
            param = Parameter(check=check(source='standby', target=['standby', 'moving']))

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

    def __init__(self, fget=None, fset=None, fget_target=None, data=None, check=None, help=None):
        """
        *fget* is a callable that is called when reading the parameter. *fset*
        is called when the parameter is written to. *fget_target* is a getter for the target value.
        *fget*, *fset*, *fget_target* must be member functions of the corresponding Parameterizable
        object.

        *data* is passed to the state check function.

        *check* is a :func:`.check` that changes states when a value
        is written to the parameter.

        *help* is a string describing the parameter in more detail.
        """
        self.name = None
        self.fget = fget
        self.fset = fset
        self.fget_target = fget_target
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

    @memoize
    def getter_name_target(self):
        if self.fget_target:
            return self.fget_target.__name__

        return '_get_target_' + self.name

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
        except KeyboardInterrupt:
            # do not scream
            LOG.debug("KeyboardInterrupt caught while getting `{}'".format(self.name))

    def __set__(self, instance, value):
        try:
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

            name = instance.__class__.__name__
            if hasattr(instance, 'name_for_log'):
                instance_name = getattr(instance, 'name_for_log')
            else:
                instance_name = _find_object_by_name(instance)
                if instance_name:
                    setattr(instance, 'name_for_log', instance_name)
            if instance_name:
                msg = "set {}::{}.{}='{}'"
                LOG.info(msg.format(name, instance_name, self.name, value))
            else:
                msg = "set {}::{}='{}'"
                LOG.info(msg.format(name, self.name, value))
        except KeyboardInterrupt:
            cancel_name = '_cancel_' + self.name
            if hasattr(instance, cancel_name):
                getattr(instance, cancel_name)()


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
        if self.fget:
            value = self.fget(instance, *self.data_args)
        else:
            try:
                value = getattr(instance, self.getter_name())(*self.data_args)
            except AccessorNotImplementedError:
                if self.default is None:
                    raise StateError('Software state must have a default value')
                value = self._value(instance)

        return value

    def __set__(self, instance, value):
        raise AttributeError('State cannot be set')

    def _value(self, instance):
        if not hasattr(instance, '_state_value'):
            setattr(instance, '_state_value', self.default)

        return getattr(instance, '_state_value')


class Selection(Parameter):

    """A :class:`.Parameter` that can take a value out of pre-defined list."""

    def __init__(self, iterable, fget=None, fset=None, help=None):
        """
        *fget*, *fset*, *data*, *transition* and *help* are identical to the
        :class:`.Parameter` constructor arguments.

        *iterable* is the list of things, that a selection can be.
        """
        super(Selection, self).__init__(fget=fget, fset=fset, help=help)
        self.iterable = iterable

    def __set__(self, instance, value):
        if value not in self.iterable:
            raise WriteAccessError('{} not in {}'.format(value, self.iterable))

        super(Selection, self).__set__(instance, value)


class Quantity(Parameter):

    """A :class:`.Parameter` associated with a unit."""

    def __init__(self, unit, fget=None, fset=None, fget_target=None, lower=None, upper=None,
                 data=None, check=None, external_lower_getter=None, external_upper_getter=None,
                 help=None):
        """
        *fget*, *fset*, *data*, *check* and *help* are identical to the
        :class:`.Parameter` constructor arguments.

        *unit* is a Pint quantity. *lower* and *upper* denote soft limits
        between the :class:`.Quantity` values can lie.
        """
        super(Quantity, self).__init__(fget=fget, fset=fset, fget_target=fget_target, data=data,
                                       check=check, help=help)
        self.unit = unit

        self.upper = upper
        self.lower = lower
        self.external_lower_getter = external_lower_getter
        self.external_upper_getter = external_upper_getter

    def convert(self, value):
        if self.unit == "delta_degC":
            return value
        else:
            return value.to(self.unit)

    def __get__(self, instance, owner):
        # If we would just call self.fset(value) we would call the method
        # defined in the base class. This is a hack (?) to call the function on
        # the instance where we actually want the function to be called.
        try:
            value = super(Quantity, self).__get__(instance, owner)

            if value is not None:
                return self.convert(value)
        except KeyboardInterrupt:
            LOG.debug("KeyboardInterrupt caught while getting `{}'".format(self.name))

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

        if lower is not None:
            if not leq(lower, value):
                msg = "{} is out of range [{}, {}]"
                raise SoftLimitError(msg.format(value, lower, upper))
        if upper is not None:
            if not leq(value, upper):
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
        table.add_row(["target_readable", self.target_readable])
        if self.target_readable:
            table.add_row(["target_value", self.get_target().join().result])
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
        return (getattr(self._instance, '_set_' + self.name) is not _setter_not_implemented or
                self._parameter.fset is not None)

    @property
    def target_readable(self):
        return (getattr(self._instance, '_get_target_' + self.name)
                is not _getter_target_not_implemented or self._parameter.fget_target is not None)

    @property
    def target(self):
        return self.get_target().join().result()

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
    def get_target(self, wait_on=None):
        """
        Get target value of this object
        :param wait_on: If not None, this must be a future which joins before reading value.
        :return:
        """
        if wait_on:
            wait_on.join()

        if self._parameter.fget_target is not None:
            return self._parameter.fget_target(self._instance)
        else:
            return getattr(self._instance, '_get_target_' + self.name)()

    def set(self, value, wait_on=None):
        """
        Set concrete *value* on the object.

        If *wait_on* is not None, it must be a future on which this method
        joins.
        """
        @async
        def execute():
            if wait_on:
                wait_on.join()

            setattr(self._instance, self.name, value)

        future = execute()
        cancel_name = '_cancel_' + self.name

        if hasattr(self._instance, cancel_name):
            future.cancel_operation = getattr(self._instance, cancel_name)

        return future

    @async
    def stash(self):
        """Save the current value internally on a growing stack.

        If the parameter is writable the current value is saved on a stack and
        to be later retrieved with :meth:`.ParameterValue.restore`.
        """
        if not self.writable:
            raise ParameterError("Parameter `{}' is not writable".format(self.name))

        if self.target_readable:
            self._saved.append(self.get_target().result())
        else:
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
        self._external_upper_getter = quantity.external_upper_getter
        self._external_lower_getter = quantity.external_lower_getter
        self._limits_locked = False

    def lock_limits(self, permanent=False):
        """Lock limits, if *permanent* is True the limits cannot be unlocked anymore."""
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
        if self.lower_user is None and self.lower_external is None:
            return None
        elif self.lower_user is None and self.lower_external is not None:
            return self.lower_external
        elif self.lower_user is not None and self.lower_external is None:
            return self.lower_user
        else:
            return np.max((self.lower_user.to(self._parameter.unit).magnitude,
                           self.lower_external.to(self._parameter.unit).magnitude),
                          axis=0) * self._parameter.unit

    @lower.setter
    def lower(self, value):
        if value is None:
            self._lower = None
            return
        self._check_limit(value)
        if self._upper is not None and value >= self._upper:
            raise ValueError('Lower limit must be lower than upper')
        self._lower = value

    @property
    def upper(self):
        if self.upper_user is None and self.upper_external is None:
            return None
        elif self.upper_user is None and self.upper_external is not None:
            return self.upper_external
        elif self.upper_user is not None and self.upper_external is None:
            return self.upper_user
        else:
            return np.min((self.upper_user.to(self._parameter.unit).magnitude,
                           self.upper_external.to(self._parameter.unit).magnitude),
                          axis=0) * self._parameter.unit

    @upper.setter
    def upper(self, value):
        if value is None:
            self._upper = None
            return
        self._check_limit(value)
        if self._lower is not None and value <= self._lower:
            raise ValueError('Upper limit must be greater than lower')
        self._upper = value

    @property
    def upper_user(self):
        return self._upper

    @property
    def lower_user(self):
        return self._lower

    @property
    def lower_external(self):
        if self._external_lower_getter is None:
            return None
        else:
            return self._external_lower_getter()

    @property
    def upper_external(self):
        if self._external_upper_getter is None:
            return None
        else:
            return self._external_upper_getter()

    @property
    def info_table(self):
        table = super(QuantityValue, self).info_table
        table.add_row(["lower", self.lower])
        table.add_row(["upper", self.upper])

        if self._external_lower_getter is not None:
            table.add_row(["lower_user", self.lower_user])
            table.add_row(["lower_external", self.lower_external])
        if self._external_upper_getter is not None:
            table.add_row(["upper_user", self.upper_user])
            table.add_row(["upper_external", self.upper_external])
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

    def _check_limit(self, value):
        """Common tasks for lower and upper before we set them."""
        if self._limits_locked:
            raise LockError('upper limit locked')
        if not _is_compatible(self._parameter.unit, value):
            raise UnitError("limit units must be compatible with `{}'".
                            format(self._parameter.unit))


class SelectionValue(ParameterValue):

    """Descriptor for :class:`.Selection` class."""

    def __init__(self, instance, selection):
        super(SelectionValue, self).__init__(instance, selection)

    @property
    def values(self):
        """Selection values."""
        return tuple(self._parameter.iterable)


class Parameterizable(object):

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

    If the parameter name does not exist, a :class:`.ParameterError` is raised.

    Each parameter value is accessible as a property. If a device has a
    position it can be read and written with::

        param.position = 0 * q.mm
        print param.position
    """

    def __init__(self):
        if not hasattr(self, '_params'):
            self._params = {}

        for base in self.__class__.__mro__:
            for attr_name, attr_type in base.__dict__.items():
                if isinstance(attr_type, Parameter):
                    attr_type.name = attr_name
                    self._install_parameter(attr_type)

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
            self._install_parameter(param)

        # Obtain the object-dot notation.
        merged_dict = dict(self.__class__.__dict__.items() + params.items())
        self.__class__ = type(self.__class__.__name__, self.__class__.__bases__, merged_dict)

    def _install_parameter(self, param):
        if isinstance(param, Quantity):
            value = QuantityValue(self, param)
        elif isinstance(param, Selection):
            value = SelectionValue(self, param)
        else:
            value = ParameterValue(self, param)

        self._params[param.name] = value

        setattr(self, 'set_' + param.name, value.set)
        setattr(self, 'get_' + param.name, value.get)

        if not hasattr(self, '_set_' + param.name):
            setattr(self, '_set_' + param.name, _setter_not_implemented)

        if not hasattr(self, '_get_' + param.name):
            setattr(self, '_get_' + param.name, _getter_not_implemented)

        if not hasattr(self, '_get_target_' + param.name):
            setattr(self, '_get_target_' + param.name, _getter_target_not_implemented)

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
