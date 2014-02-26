# -*- coding: utf-8 -*-
"""Core module Parameters"""
import numpy as np
import logging
import six
import collections
import functools
import inspect
import types
from concert.helpers import memoize
from concert.async import async, wait


LOG = logging.getLogger(__name__)


def identity(x):
    return x


class TransitionNotAllowed(Exception):
    pass


class StateError(Exception):

    """Raised in check functions of state transitions of devices."""

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


class State(object):

    """
    Finite state machine.

    Use this on a class, to keep some sort of known state. In order to enforce
    restrictions, you would decorate methods on the class with
    :func:`.transition`::

        class SomeObject(object):

            state = State(default='standby')

            @transition(source='*', target='moving')
            def move(self):
                pass

    Accessing the state variable will return the current state value, i.e.::

        obj = SomeObject()
        assert obj.state == 'standby'

    The state cannot be set explicitly by::

        obj.state = 'some_state'

    but the object needs to provide methods which transition out of
    states, the same holds for transitioning out of error states.
    """

    def __init__(self, default=None):
        self.default = default

    def __get__(self, instance, owner):
        return self._value(instance)

    def __set__(self, instance, value):
        raise AttributeError('State cannot be set')

    def _value(self, instance):
        if not hasattr(instance, '_state_value'):
            setattr(instance, '_state_value', self.default)

        return getattr(instance, '_state_value')


def transition(source='*', target=None, immediate=None, check=None):
    """
    Decorates a method that triggers state transitions.

    *source* denotes the source state that must be present at the time of
    invoking the decorated method. *target* is the state that the state object
    will be after successful completion of the method or a list of possible
    target states. *immediate* is an optional state that will be set during
    execution of the method.

    *check* is a callable that will be called to determine the actual state in
    case *target* is a list of possible target states. Hence, when *check* is
    called it must return one of those target states.
    """
    def wrapped(func):
        transitions = collections.defaultdict(list)

        sources = [source] if isinstance(source, str) else source
        targets = [target] if isinstance(target, str) else target

        if immediate:
            sources.append(immediate)
            targets.append(immediate)

        for s in sources:
            transitions[s] = targets

        def _value(instance):
            if not hasattr(instance, '_state_value'):
                setattr(instance, '_state_value', instance.state)
            return instance.state

        def try_transition(target, instance, *args, **kwargs):
            current = _value(instance)
            succ = transitions.get(current, transitions.get('*', None))

            if not succ:
                msg = "Cannot transition from `{}' to `{}'".format(current, target)
                raise TransitionNotAllowed(msg)

        @functools.wraps(func)
        def call_func(instance, *args, **kwargs):
            current = _value(instance)

            if current not in sources and '*' not in sources:
                msg = "`{}' not in `{}'".format(source, sources)
                raise TransitionNotAllowed(msg)

            if immediate:
                # Since it was listed by the user, the transition must exist so no check required.
                setattr(instance, '_state_value', immediate)

            try:
                # If there is an edge to the desired transition from source or immediate
                # we execute the actual function
                try_transition(target, instance)

                if isinstance(func, types.MethodType):
                    result = func(*args, **kwargs)
                else:
                    result = func(instance, *args, **kwargs)

                # The final state can come from the device itself (more possible target states)
                final = getattr(instance, check.__name__)() if isinstance(target, list) else target
                # We check the actual state after the function execution before we do the final
                # transition
                try_transition(final, instance)
                setattr(instance, '_state_value', final)
            except StateError as error:
                setattr(instance, '_state_value', error.state)
                raise error

            return result

        return call_func

    return wrapped


class Parameter(object):

    """A parameter with getter and setter.

    Parameters are similar to normal Python properties and can additionally
    trigger state transitions. If *fget* or *fset* is not given, you must
    implement the accessor functions named `_set_name` and `_get_name`::

        from concert.base import Parameter, State

        class SomeClass(object):

            state = State(default='standby')

            def actual(self):
                return 'moving'

            param = Parameter(transition=transition(source='standby',
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

    def __init__(self, fget=None, fset=None, data=None, transition=None, help=None):
        """
        *fget* is a callable that is called when reading the parameter. *fset*
        is called when the parameter is written to.

        *data* is passed to the state transition function.

        *transition* is a :func:`.transition` that changes states when a value
        is written to the parameter.

        *help* is a string describing the parameter in more detail.
        """
        self.name = None
        self.fget = fget
        self.fset = fset
        self.data_args = (data,) if data is not None else ()
        self.transition = transition
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
        except NotImplementedError:
            raise ReadAccessError(self.name)

    def __set__(self, instance, value):
        def log_access(what):
            """Log access."""
            msg = "{}: {}: {}='{}'"
            name = instance.__class__.__name__
            LOG.info(msg.format(name, what, self.name, value))

        log_access('try')

        if self.fset:
            self.fset(instance, value, *self.data_args)
        else:
            func = getattr(instance, '_set_' + self.name)

            if self.transition and not hasattr(func, '_is_transitioned'):
                func = self.transition(func)
                func._is_transitioned = True
                setattr(instance, '_set_' + self.name, func)

            try:
                if isinstance(func, types.MethodType):
                    func(value, *self.data_args)
                else:
                    func(instance, value, *self.data_args)
            except NotImplementedError:
                raise WriteAccessError(self.name)

        log_access('set')


class Quantity(Parameter):

    """A :class:`.Parameter` associated with a unit."""

    def __init__(self, unit, fget=None, fset=None, lower=None, upper=None,
                 conversion=identity, data=None, transition=None, help=None):
        """
        *fget*, *fset*, *data*, *transition* and *help* are identical to the
        :class:`.Parameter` constructor arguments.

        *unit* is a Pint quantity. *lower* and *upper* denote soft limits
        between the :class:`.Quantity` values can lie.

        *conversion* is a callable that is run whenever the value is read and
        written. By default, *conversion* is the identity f(x) = x.
        """
        super(Quantity, self).__init__(fget=fget, fset=fset, data=data,
                                       transition=transition, help=help)
        self.unit = unit
        self.default_conversion = conversion

        self.upper = upper if upper is not None else float('Inf')
        self.lower = lower if lower is not None else -float('Inf')

        if upper is None:
            self.upper = self.upper * unit

        if lower is None:
            self.lower = self.lower * unit

    def is_compatible(self, value):
        try:
            self.unit + value
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
        if not self.is_compatible(value):
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

        converted = self.convert_to(instance, value)
        super(Quantity, self).__set__(instance, converted)


def quantity(unit=None, lower=None, upper=None, conversion=identity,
             data=None, transition=None, help=None):
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
                        conversion=conversion, data=data,
                        transition=transition, help=doc)

    return wrapper

class ParameterValue(object):

    """Value object of a :class:`.Parameter`."""

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

    def __repr__(self):
        return str(self._parameter)

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
        to be later retrieved with :meth:`.ParameterValue.restore`.
        """
        self._saved.append(self.get().result())

    def restore(self):
        """Restore the last value saved with :meth:`.ParameterValue.stash`.

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

        # If self would be Stateful, I'd feel better ...
        if hasattr(self, 'state'):
            table.add_row(['state', self.state])

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
        :meth:`.Parameterizable.restore`.

        The values are stored on a stacked, hence subsequent saved states can
        be restored one by one.
        """
        wait((param.stash() for param in self))

    @async
    def restore(self):
        """Restore all parameters saved with :meth:`.Parameterizable.stash`."""
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
