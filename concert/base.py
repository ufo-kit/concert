# -*- coding: utf-8 -*-
"""Core module Parameters"""
import asyncio
import numpy as np
import logging
import functools
import inspect
import types
from concert.helpers import memoize
from concert.coroutines.base import background, get_event_loop, run_in_loop, wait_until
from concert.quantities import q


LOG = logging.getLogger(__name__)
_RUN_IN_LOOP_ERR_TMPL = "Someone is trying to use `{}' " \
                        "inside `async def' code which is not allowed, please " \
                        "use `await {}' instead"


def identity(x):
    return x


def _setter_not_implemented(value, *args):
    raise AccessorNotImplementedError


def _getter_not_implemented(*args):
    raise AccessorNotImplementedError


def _getter_target_not_implemented(*args):
    raise AccessorNotImplementedError


async def _execute_func(func, instance, *args, **kwargs):
    """Execute *func* irrespective of whether it is a function or a method. *instance*
    is discarded if *func* is a function, otherwise it is used as a first real argument.
    """
    if isinstance(func, types.MethodType):
        result = await func(*args, **kwargs)
    else:
        result = await func(instance, *args, **kwargs)

    return result


def _find_object_by_name(instance):
    """Find variable name by *instance*. This is supposed to be used only in the
    :meth:`.Parameter.__set__`.
    """
    def _is_name_ok(instance_name):
        return instance_name and instance_name not in ['instance', 'self']

    def _find_in_dict(dictionary):
        for (obj_name, obj) in list(dictionary.items()):
            if obj is instance:
                return obj_name

    instance_name = None

    try:
        ipython = get_ipython()
        instance_name = _find_in_dict(ipython.user_ns)
        if instance_name:
            return instance_name
    except NameError:
        # Not in IPython
        pass

    frames = inspect.stack()
    try:
        # Skip us and ParameterValue.set
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
            if _is_name_ok(instance_name):
                break
    finally:
        # Cleanup as python docs suggest
        del frames

    if not instance_name:
        # Try coroutines
        tasks = asyncio.all_tasks(loop=asyncio.get_running_loop())
        for task in tasks:
            # Python 3.7 compatibility, later task.get_coro()
            coro = task._coro
            while coro:
                instance_name = _find_in_dict(inspect.getcoroutinelocals(coro))
                if _is_name_ok(instance_name):
                    return instance_name
                # task._coro can be awaiting another coroutine, so let's recurse
                coro = coro.cr_await if inspect.iscoroutine(coro.cr_await) else None

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

    """Raised when user tries to read a parameter that cannot be read."""

    def __init__(self, parameter):
        LOG.warning("Invalid read access on {0}".format(parameter))
        msg = "parameter `{0}' cannot be read".format(parameter)
        super(ReadAccessError, self).__init__(msg)


class TargetAccessError(Exception):

    """Raised when user tries to read parameter's target value that cannot be read."""

    def __init__(self, parameter):
        LOG.warning("Invalid read access on {0}".format(parameter))
        msg = "parameter's `{0}' target value cannot be read".format(parameter)
        super(TargetAccessError, self).__init__(msg)


class WriteAccessError(Exception):

    """Raised when user tries to read a parameter that cannot be read."""

    def __init__(self, parameter):
        LOG.warning("Invalid write access on {0}".format(parameter))
        msg = "parameter `{0}' cannot be written".format(parameter)
        super(WriteAccessError, self).__init__(msg)


class SelectionError(Exception):

    """Raised when user tries to set selection parameter to unknown value."""

    def __init__(self, parameter, value, values):
        LOG.warning("Invalid write access on {0}".format(parameter))
        msg = "Value `{}' not in `{}'".format(value, values)
        super(SelectionError, self).__init__(msg)


class LockError(Exception):

    """Raised when parameter is locked."""

    pass


def transition(immediate=None, target=None):
    """Change software state of a device to *immediate*. After the function execution finishes
    change the state to *target*. On :py:class:`.asyncio.CancelledError`, state is set to *target*
    and cleanup logic must take place in the callable to be wrapped.
    """
    def wrapped(func):
        @functools.wraps(func)
        async def call_func(instance, *args, **kwargs):
            if 'state' not in instance:
                raise FSMError('Changing state requires state parameter')

            # Store the original in case target is None
            target_state = target if target else await instance['state'].get()

            if immediate:
                setattr(instance, '_state_value', immediate)

            try:
                result = await _execute_func(func, instance, *args, **kwargs)
                setattr(instance, '_state_value', target_state)
            except StateError as error:
                setattr(instance, '_state_value', error.state)
                raise error
            except asyncio.CancelledError:
                # If *func* is cancelled, *func* must make sure that all cleanup except state update
                # happens. If even the state should have some special value, then do not use
                # *transition* at all and implement everything in *func*.
                setattr(instance, '_state_value', target_state)
                raise

            return result

        return call_func

    return wrapped


def check(source='*', target='*'):
    """
    Decorates a method for checking the device state.

    *source* denotes the source state that must be present at the time of
    invoking the decorated method. *target* is the state that the state object
    will be after successful completion of the method or a list of possible
    target states.
    """
    async def check_now(instance, allowed_states, state_str):
        state = await instance['state'].get()
        if state not in allowed_states and '*' not in allowed_states:
            raise TransitionNotAllowed(f"{state_str} state `{state}' not in `{allowed_states}'")

    def wrapped(func):
        sources = [source] if isinstance(source, str) else source
        targets = [target] if isinstance(target, str) else target

        @functools.wraps(func)
        async def call_func(instance, *args, **kwargs):
            if 'state' not in instance:
                raise FSMError('Transitioning requires state parameter')

            await check_now(instance, sources, 'Current')
            result = await _execute_func(func, instance, *args, **kwargs)
            await check_now(instance, targets, 'Final')

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

    @memoize
    def get_getter(self, instance):
        if self.fget:
            getter = functools.partial(self.fget, instance)
        else:
            getter = getattr(instance, self.getter_name())

        return getter

    @memoize
    def get_target_getter(self, instance):
        if self.fget_target:
            target_getter = functools.partial(self.fget_target, instance)
        else:
            target_getter = getattr(instance, self.getter_name_target())

        return target_getter

    @memoize
    def get_setter(self, instance):
        if self.fset:
            # If check is supplied to the Parameter do not apply partial because it will be applied
            # below
            setter = self.fset if self.check else functools.partial(self.fset, instance)
        else:
            setter = getattr(instance, self.setter_name())

        if self.check:
            setter = self.check(setter)
            # Check wraps a method into a function which expects the first argument to be instance
            # again, thus wrap it by functools.partial
            setter = functools.partial(setter, instance)

        return setter

    def __repr__(self):
        return str(self.help)

    def __get__(self, instance, owner):
        return run_in_loop(
            instance[self.name].get(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'obj.param',
                'obj.get_param()'
            )
        )

    def __set__(self, instance, value):
        run_in_loop(
            instance[self.name].set(value),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'obj.param = value',
                'obj.set_param(value)'
            )
        )


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

    def __set__(self, instance, value):
        raise AttributeError('State cannot be set')

    def _value(self, instance):
        if not hasattr(instance, '_state_value'):
            setattr(instance, '_state_value', self.default)

        return getattr(instance, '_state_value')


class Selection(Parameter):

    """A :class:`.Parameter` that can take a value out of pre-defined list."""

    def __init__(self, iterable, fget=None, fset=None, check=None, help=None):
        """
        *fget*, *fset*, *check* and *help* are identical to the :class:`.Parameter` constructor
        arguments.

        *iterable* is the list of things, that a selection can be.
        """
        super(Selection, self).__init__(fget=fget, fset=fset, check=check, help=help)
        self.iterable = iterable


class Quantity(Parameter):

    """A :class:`.Parameter` associated with a unit."""

    def __init__(self, unit, fget=None, fset=None, fget_target=None, lower=None, upper=None,
                 data=None, check=None, external_lower_getter=None, external_upper_getter=None,
                 user_lower_getter=None, user_lower_setter=None, user_upper_getter=None,
                 user_upper_setter=None, help=None):
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
        self.user_lower_getter = user_lower_getter
        self.user_lower_setter = user_lower_setter
        self.user_upper_getter = user_upper_getter
        self.user_upper_setter = user_upper_setter

    def convert(self, value):
        return value.to(self.unit)


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
        self._lock = asyncio.Lock()
        self._locked = False
        self._instance = instance
        self._parameter = parameter
        self._saved = []

    async def __aenter__(self):
        await self._lock.acquire()

        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._lock.release()

    def __lt__(self, other):
        return self._parameter.name < other._parameter.name

    def __repr__(self):
        if get_event_loop().is_running():
            return repr(self._parameter)
        else:
            return run_in_loop(self.info_table).get_string()

    @property
    async def info_table(self):
        from concert.session.utils import get_default_table
        locked = "yes" if self.locked else "no"
        table = get_default_table(["attribute", "value"])
        table.header = False
        table.border = False
        table.add_row(["info", self._parameter.help])
        table.add_row(["locked", locked])
        table.add_row(["target_readable", self.target_readable])
        if self.target_readable:
            table.add_row(["target_value", await self.get_target()])
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
        return (getattr(self._instance, '_set_' + self.name) is not _setter_not_implemented
                or self._parameter.fset is not None)

    @property
    def target_readable(self):
        return (getattr(self._instance, '_get_target_' + self.name)
                is not _getter_target_not_implemented or self._parameter.fget_target is not None)

    @property
    def target(self):
        return run_in_loop(
            self.get_target(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'parameter.target',
                'parameter.get_target()'
            )
        )

    @background
    async def get(self, wait_on=None):
        """
        Get coroutine obtaining the concrete *value* of this object.

        If *wait_on* is not None, it must be an awaitable on which this method waits.
        """
        if wait_on:
            await wait_on

        try:
            getter = self._parameter.get_getter(self._instance)
            return await getter(*self._parameter.data_args)
        except asyncio.CancelledError:
            LOG.debug('getter cancelled %s', self._parameter.name)
        except AccessorNotImplementedError:
            raise ReadAccessError(self.name)

    @background
    async def get_target(self, wait_on=None):
        """
        Get coroutine obtaining target value of this object.

        If *wait_on* is not None, it must be an awaitable on which this method waits.
        """
        if wait_on:
            await wait_on

        try:
            target_getter = self._parameter.get_target_getter(self._instance)
            return await target_getter()
        except AccessorNotImplementedError:
            raise TargetAccessError(self.name)

    @background
    async def set(self, value, wait_on=None):
        """
        Set concrete *value* on the object.

        If *wait_on* is not None, it must be an awaitable on which this method waits.
        """
        if wait_on:
            await wait_on

        if self.locked:
            raise LockError("Parameter `{}' is locked for writing".format(self._parameter.name))

        try:
            setter = self._parameter.get_setter(self._instance)
            await setter(value, *self._parameter.data_args)
        except AccessorNotImplementedError:
            raise WriteAccessError(self.name)

        name = self._instance.__class__.__name__
        if hasattr(self._instance, 'name_for_log'):
            instance_name = getattr(self._instance, 'name_for_log')
        else:
            instance_name = _find_object_by_name(self._instance)
            if instance_name:
                setattr(self._instance, 'name_for_log', instance_name)
        if instance_name:
            msg = "set {}::{}.{}='{}'"
            LOG.info(msg.format(name, instance_name, self._parameter.name, value))
        else:
            msg = "set {}::{}='{}'"
            LOG.info(msg.format(name, self._parameter.name, value))

    @background
    async def stash(self):
        """Save the current value internally on a growing stack.

        If the parameter is writable the current value is saved on a stack and
        to be later retrieved with :meth:`.ParameterValue.restore`.
        """
        if not self.writable:
            raise ParameterError("Parameter `{}' is not writable".format(self.name))

        if self.target_readable:
            self._saved.append(await self.get_target())
        else:
            self._saved.append(await self.get())

    @background
    async def restore(self):
        """Restore the last value saved with :meth:`.ParameterValue.stash`.

        If the parameter can only be read or no value has been saved, this
        operation does nothing.
        """
        if not self.writable:
            raise ParameterError("Parameter `{}' is not writable".format(self.name))

        if self._saved:
            val = self._saved.pop()
            await self.set(val)

    @property
    def locked(self):
        """Return True if the parameter is locked for writing."""
        return self._locked

    def lock(self, permanent=False):
        """Lock parameter for writing. If *permament* is True the parameter
        cannot be unlocked anymore.
        """
        def unlock_not_allowed():
            raise LockError("Parameter `{}' cannot be unlocked".format(self._parameter.name))

        self._locked = True
        if permanent:
            self.unlock = unlock_not_allowed

    def unlock(self):
        """Unlock parameter for writing."""
        self._locked = False

    @background
    async def wait(self, value, sleep_time=1e-1 * q.s, timeout=None):
        """Wait until the parameter value is *value*. *sleep_time* is the time to sleep
        between consecutive checks. *timeout* specifies the maximum waiting time.
        """
        async def condition():
            return await self.get() == value

        await wait_until(condition, sleep_time=sleep_time, timeout=timeout)


class StateValue(ParameterValue):

    """Special :class:`.ParameterValue` implementing state parameter."""

    @background
    async def get(self, wait_on=None):
        if wait_on:
            await wait_on

        try:
            getter = self._parameter.get_getter(self._instance)
            return await getter(*self._parameter.data_args)
        except AccessorNotImplementedError:
            if not self._parameter.fget:
                if self._parameter.default is None:
                    raise FSMError('Software state must have a default value')
                if not hasattr(self._instance, '_state_value'):
                    self._instance._state_value = self._parameter.default
                return self._instance._state_value
            else:
                raise ReadAccessError(self.name)


class QuantityValue(ParameterValue):

    def __init__(self, instance, quantity):
        super(QuantityValue, self).__init__(instance, quantity)
        self._lower = quantity.lower
        self._upper = quantity.upper
        self._external_upper_getter = quantity.external_upper_getter
        self._external_lower_getter = quantity.external_lower_getter
        self._user_lower_getter = quantity.user_lower_getter
        self._user_lower_setter = quantity.user_lower_setter
        self._user_upper_getter = quantity.user_upper_getter
        self._user_upper_setter = quantity.user_upper_setter
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
        return run_in_loop(
            self.get_lower(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.lower',
                'quantity.get_lower()'
            )
        )

    @lower.setter
    def lower(self, value):
        run_in_loop(
            self.set_lower(value),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.lower = value',
                'quantity.set_lower(value)'
            )
        )

    @background
    async def get_lower(self):
        lower_external = await self.get_lower_external()
        lower_user = await self.get_lower_user()
        if lower_user is None and lower_external is None:
            return None
        elif lower_user is None and lower_external is not None:
            return lower_external
        elif lower_user is not None and lower_external is None:
            return lower_user
        else:
            return np.max((lower_user.to(self._parameter.unit).magnitude,
                           lower_external.to(self._parameter.unit).magnitude),
                          axis=0) * self._parameter.unit

    @background
    async def set_lower(self, value):
        self._check_limit(value)
        upper = await self.get_upper()
        if value is not None and upper is not None and value >= upper:
            raise ValueError('Lower limit must be lower than upper')
        if self._user_lower_setter:
            await self._user_lower_setter(value)
        else:
            self._lower = value

    @property
    def upper(self):
        return run_in_loop(
            self.get_upper(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.upper',
                'quantity.get_upper()'
            )
        )

    @upper.setter
    def upper(self, value):
        run_in_loop(
            self.set_upper(value),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.upper = value',
                'quantity.set_upper(value)'
            )
        )

    @background
    async def get_upper(self):
        upper_external = await self.get_upper_external()
        upper_user = await self.get_upper_user()
        if upper_user is None and upper_external is None:
            return None
        elif upper_user is None and upper_external is not None:
            return upper_external
        elif upper_user is not None and upper_external is None:
            return upper_user
        else:
            return np.min((upper_user.to(self._parameter.unit).magnitude,
                           upper_external.to(self._parameter.unit).magnitude),
                          axis=0) * self._parameter.unit

    @background
    async def set_upper(self, value):
        self._check_limit(value)
        lower = await self.get_lower()
        if value is not None and lower is not None and value <= lower:
            raise ValueError('Upper limit must be greater than lower')
        if self._user_upper_setter:
            await self._user_upper_setter(value)
        else:
            self._upper = value

    @property
    def upper_user(self):
        return run_in_loop(
            self.get_upper_user(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.upper_user',
                'quantity.get_upper_user()'
            )
        )

    @background
    async def get_upper_user(self):
        if self._user_upper_getter is None:
            return self._upper
        else:
            return await self._user_upper_getter()

    @property
    def lower_user(self):
        return run_in_loop(
            self.get_lower_user(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.lower_user',
                'quantity.get_lower_user()'
            )
        )

    @background
    async def get_lower_user(self):
        if self._user_lower_getter is None:
            return self._lower
        else:
            return await self._user_lower_getter()

    @property
    def lower_external(self):
        return run_in_loop(
            self.get_lower_external(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.lower_external',
                'quantity.get_lower_external()'
            )
        )

    @background
    async def get_lower_external(self):
        if self._external_lower_getter is None:
            return None
        else:
            return await self._external_lower_getter()

    @property
    def upper_external(self):
        return run_in_loop(
            self.get_upper_external(),
            error_msg_if_running=_RUN_IN_LOOP_ERR_TMPL.format(
                'quantity.upper_external',
                'quantity.get_upper_external()'
            )
        )

    @background
    async def get_upper_external(self):
        if self._external_upper_getter is None:
            return None
        else:
            return await self._external_upper_getter()

    @property
    async def info_table(self):
        table = await super(QuantityValue, self).info_table
        table.add_row(["lower", await self.get_lower()])
        table.add_row(["upper", await self.get_upper()])

        if self._external_lower_getter is not None:
            table.add_row(["lower_user", await self.get_lower_user()])
            table.add_row(["lower_external", await self.get_lower_external()])
        if self._external_upper_getter is not None:
            table.add_row(["upper_user", await self.get_upper_user()])
            table.add_row(["upper_external", await self.get_upper_external()])
        return table

    @property
    def unit(self):
        return self._parameter.unit

    @background
    async def get(self, wait_on=None):
        value = await super(QuantityValue, self).get(wait_on=wait_on)

        if value is not None:
            return self._parameter.convert(value)

    @background
    async def set(self, value, wait_on=None):
        """
        Set concrete *value* on the object.

        If *wait_on* is not None, it must be an awaitable on which this method waits.
        """
        if wait_on:
            await wait_on

        if not self.unit.is_compatible_with(value):
            msg = "{} of {} can only receive values of unit {} but got {}"
            raise UnitError(
                msg.format(self._parameter.name, type(self._instance), self.unit, value))

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

        lower = await self.get_lower()
        upper = await self.get_upper()
        if lower is not None:
            if not leq(lower, value):
                msg = "{} is out of range [{}, {}]"
                raise SoftLimitError(msg.format(value, lower, upper))
        if upper is not None:
            if not leq(value, upper):
                msg = "{} is out of range [{}, {}]"
                raise SoftLimitError(msg.format(value, lower, upper))

        converted = self._parameter.convert(value)
        await super(QuantityValue, self).set(converted, wait_on=wait_on)

    @background
    async def wait(self, value, eps=None, sleep_time=1e-1 * q.s, timeout=None):
        """Wait until the parameter value is *value*. *eps* is the allowed discrepancy between the
        actual value and *value*. *sleep_time* is the time to sleep between consecutive checks.
        *timeout* specifies the maximum waiting time.
        """
        async def condition():
            """Take care of rountind errors"""
            diff = await self.get() - value
            if eps is not None:
                diff = np.abs(diff.to(eps.units))
                if diff < eps:
                    diff = 0

            return diff == 0

        await wait_until(condition, sleep_time=sleep_time, timeout=timeout)

    def _check_limit(self, value):
        """Common tasks for lower and upper before we set them."""
        if value is None:
            return
        if self._limits_locked:
            raise LockError('upper limit locked')
        if not self.unit.is_compatible_with(value):
            raise UnitError("limit units must be compatible with `{}'".format(self.unit))


class SelectionValue(ParameterValue):

    """Descriptor for :class:`.Selection` class."""

    def __init__(self, instance, selection):
        super(SelectionValue, self).__init__(instance, selection)

    @background
    async def set(self, value, wait_on=None):
        """
        Set concrete *value* on the object.

        If *wait_on* is not None, it must be an awaitable on which this method waits.
        """
        if value not in self._parameter.iterable:
            raise SelectionError(self._parameter.name, value, self._parameter.iterable)

        await super(SelectionValue, self).set(value, wait_on=wait_on)

    @property
    def values(self):
        """Selection values."""
        return tuple(self._parameter.iterable)


class AsyncType(type):

    """
    Metaclass which allows an awaitable constructor `__ainit__(...)' instead of `__init__(...)' and
    thus enables async object initialization `obj = await Class(...)'. `__ainit__(...)` must return
    an awaitable and its result must be None. To actually make async objects use the
    :class:`.AsyncObject` class.

    By having a new method for async initialization we clearly separate the async and non-async
    worlds. In case of mixed async and non-async multiple inheritance, it is the responsibility of
    *Class* to call the predecessors of both parts, e.g. by `await super().__ainit__(...)' and
    `super().__init__(...)'. It is forbidden to have an `__init__(...)' in a class with
    `__ainit__(...)', so the calls to *super()* will be unambiguous with respect to the two
    constructor types and MRO will work as expected.
    """

    def __new__(meta, name, bases, dct):
        # Create new class *name* and make sure it does not implement __init__.
        if '__init__' in dct:
            raise TypeError(f"AsyncObject `{name}' must define __ainit__ instead of __init__")
        cls = super(AsyncType, meta).__new__(meta, name, bases, dct)
        if hasattr(cls, '__ainit__'):
            # Copy the signature of __ainit__ to *cls* so that helper functions which pick up on
            # constructor attributes work
            cls.__signature__ = inspect.Signature.from_callable(cls.__ainit__)

        return cls

    async def __call__(cls, *args, **kwargs):
        # Create an instance of class *cls*. Do almost the same as type.__call__
        # (Objects/typeobject.c:type_call in CPython) but instead of calling the constructor
        # __init__, call the async constructor __ainit__, i.e. __init__ is never called becuase
        # calling both __init__ and __ainit__ with the same *args* and *kwargs* may be undesirable.
        # Moreover, everything which can be constructed in __init__ can also be constructed in
        # __ainit__.
        obj = cls.__new__(cls, *args, **kwargs)

        if isinstance(obj, cls) and hasattr(cls, '__ainit__'):
            # If cls.__new__ returns and instance of cls and there is an __ainit__ method, call it
            # and make sure it does not return anything
            coro = obj.__ainit__(*args, **kwargs)
            if not inspect.isawaitable(coro):
                raise TypeError(
                    f"__ainit__() must return an awaitable, not `{type(coro).__name__}'"
                )
            result = await coro
            if result is not None:
                raise TypeError(
                    f"__ainit__() should return None, not `{type(result).__name__}'"
                )

        return obj


class AsyncObject(metaclass=AsyncType):
    """Root of all classes with async def __ainit__()."""

    def __new__(cls, *args, **kwargs):
        # Create an instance of class *cls* by calling object.__new__(*cls*) but do the argument
        # check here because we cannot re-use object's __new__ (Objects/typeobject.c:object_new in
        # CPython) becuase it checks for normal __init__ constructor and not __ainit__. If *cls*
        # re-defines __new__, it must call AsyncObject.__new__ without any arguments (*cls* is
        # responsible for consuming them). If *cls* does not re-define __new__ and defines
        # __ainit__, potentially wrong arguments will be handled when the type tries to call
        # __ainit__; if __ainit__ is not re-defined then there cannot be arguments because
        # AsyncObject.__ainit__ takes no arguments.
        if args or kwargs:
            if cls.__new__ != AsyncObject.__new__:
                raise TypeError(
                    "AsyncObject.__new__() takes exactly one argument (the type to instantiate)"
                )
            if cls.__ainit__ == AsyncObject.__ainit__:
                raise TypeError(f"{cls.__name__} takes no arguments")

        return object.__new__(cls)

    async def __ainit__(self):
        pass


class Parameterizable(AsyncObject):

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

    async def __ainit__(self):
        if not hasattr(self, '_params'):
            self._params = {}

        for base in self.__class__.__mro__:
            for attr_name, attr_type in list(base.__dict__.items()):
                if isinstance(attr_type, Parameter):
                    attr_type.name = attr_name
                    self._install_parameter(attr_type)

    def __str__(self):
        if get_event_loop().is_running():
            return super().__str__()
        else:
            return run_in_loop(self.info_table).get_string(sortby="Parameter")

    def __repr__(self):
        if get_event_loop().is_running():
            return super().__repr__()
        else:
            return '\n'.join([super(Parameterizable, self).__repr__(), str(self)])

    def __iter__(self):
        for param in sorted(self._params.values()):
            yield param

    def __getitem__(self, param):
        if param not in self._params:
            raise ParameterError(param)

        return self._params[param]

    def __contains__(self, key):
        return key in self._params

    @property
    async def info_table(self):
        from concert.session.utils import get_default_table

        table = get_default_table(["Parameter", "Value"])
        table.border = False

        for param in self:
            try:
                value = str(await param.get())
            except ReadAccessError:
                value = 'N/A'
            table.add_row([param.name, value])

        return table

    def install_parameters(self, params):
        """Install parameters at run-time.

        *params* is a dictionary mapping parameter names to :class:`.Parameter`
        objects.
        """
        for name, param in list(params.items()):
            param.name = name
            self._install_parameter(param)

        # Obtain the object-dot notation.
        merged_dict = dict(list(self.__class__.__dict__.items()) + list(params.items()))
        self.__class__ = type(self.__class__.__name__, self.__class__.__bases__, merged_dict)

    def _install_parameter(self, param):
        if isinstance(param, Quantity):
            value = QuantityValue(self, param)
        elif isinstance(param, Selection):
            value = SelectionValue(self, param)
        elif isinstance(param, State):
            value = StateValue(self, param)
        else:
            value = ParameterValue(self, param)

        self._params[param.name] = value

        getter_name = 'get_' + param.name
        setter_name = 'set_' + param.name
        target_getter_name = 'get_target_' + param.name
        if getter_name not in self.__class__.__dict__:
            def get_parameter(instance, wait_on=None):
                return instance[param.name].get(wait_on=wait_on)
            setattr(self.__class__, getter_name, get_parameter)
        if setter_name not in self.__class__.__dict__:
            def set_parameter(instance, parameter_value, wait_on=None):
                return instance[param.name].set(parameter_value, wait_on=wait_on)
            setattr(self.__class__, setter_name, set_parameter)
        if target_getter_name not in self.__class__.__dict__:
            def get_target_parameter(instance, wait_on=None):
                return instance[param.name].get_target(wait_on=wait_on)
            setattr(self.__class__, target_getter_name, get_target_parameter)

        if not hasattr(self, '_set_' + param.name):
            setattr(self, '_set_' + param.name, _setter_not_implemented)

        if not hasattr(self, '_get_' + param.name):
            setattr(self, '_get_' + param.name, _getter_not_implemented)

        if not hasattr(self, '_get_target_' + param.name):
            setattr(self, '_get_target_' + param.name, _getter_target_not_implemented)

    @background
    async def stash(self):
        """
        Save all writable parameters that can be restored with
        :meth:`.Parameterizable.restore`.

        The values are stored on a stacked, hence subsequent saved states can
        be restored one by one.
        """
        await asyncio.gather(*(param.stash() for param in self if param.writable))

    @background
    async def restore(self):
        """Restore all parameters saved with :meth:`.Parameterizable.stash`."""
        await asyncio.gather(*(param.restore() for param in self if param.writable))

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
