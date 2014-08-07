import time
import inspect
from concert.quantities import q


class Command(object):

    """Command class for the CLI script"""

    def __init__(self, name, opts):
        """
        Command objects are loaded at run-time and injected into Concert's
        command parser.

        *name* denotes the name of the sub-command parser, e.g. "mv" for the
        MoveCommand. *opts* must be an argparse-compatible dictionary
        of command options.
        """
        self.name = name
        self.opts = opts

    def run(self, *args, **kwargs):
        """Run the command"""
        raise NotImplementedError


class Bunch(object):

    """Encapsulate a list or dictionary to provide attribute-like access.

    Common use cases look like this::

        d = {'foo': 123, 'bar': 'baz'}
        b = Bunch(d)
        print(b.foo)
        >>> 123

        l = ['foo', 'bar']
        b = Bunch(l)
        print(b.foo)
        >>> 'foo'
    """

    def __init__(self, values):
        if isinstance(values, list):
            values = dict(zip(values, values))
        self.__dict__.update(values)


def memoize(func):
    """
    Memoize the result of *func*.

    Remember the result of *func* depending on its arguments. Note, that this
    requires that the function is free from any side effects, e.g. returns the
    same value given the same arguments.
    """
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]

        result = func(*args)
        memo[args] = result
        return result

    return wrapper


def busy_wait(condition, sleep_time=1e-1 * q.s, timeout=None):
    """Busy wait until a callable *condition* returns True. *sleep_time* is the time to sleep
    between consecutive checks of *condition*. If *timeout* is given and the *condition* doesn't
    return True within the time specified by it a :class:`.WaitingError` is raised.
    """
    sleep_time = sleep_time.to(q.s).magnitude
    if timeout:
        start = time.time()
        timeout = timeout.to(q.s).magnitude

    while not condition():
        if timeout and time.time() - start > timeout:
            raise WaitError('Waiting timed out')
        time.sleep(sleep_time)


class WaitError(Exception):

    """Raised on busy waiting timeouts"""
    pass


class _Structure(object):

    def __init__(self, func, e_args, f_args, f_defaults, e_keywords):
        self.func = func
        self.e_args = e_args
        self.f_args = f_args
        self.f_defaults = f_defaults
        self.outputs = e_keywords['output']
        self.e_keywords = e_keywords
        self._isfunction = True
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        self._check_args(*args, **kwargs)
        return self.func(*args, **kwargs)

    def _check_args(self, *args, **kwargs):
        for i, key in enumerate(args):
            if i < len(self.e_args):
                self._check_type_correctness(
                    self.f_args[i],
                    self.e_args[i],
                    args[i])
            else:
                self._check_type_correctness(
                    self.f_args[i],
                    self.e_keywords[self.f_args[i]],
                    args[i])
        for i, key in enumerate(kwargs):
            if kwargs[key] is not None:
                self._check_type_correctness(
                    key,
                    self.e_keywords[key],
                    kwargs[key])

    def _check_type_correctness(self, arg_name, expected, given):
        from concert.devices.base import Device
        if expected is not None:
            if isinstance(expected, Numeric):
                self._check_numeric(arg_name, expected, given)

            elif issubclass(expected, Device) and not isinstance(given, expected):
                raise TypeError(
                    'Sorry, argument "{}" expected to get {}, but got {}'.format(
                        arg_name,
                        expected.__name__,
                        given.__class__.__name__))

    def _check_numeric(self, arg_name, expected, given):
        if (expected.units is not None) ^ hasattr(given, 'units'):
            raise TypeError(
                'Sorry, argument "{}" expected to get value with unit {}, but got {}'.format(
                    arg_name,
                    expected.units,
                    given))

        elif expected.units is not None:
            e_units = expected.units.to_base_units().units
            if not e_units == given.to_base_units().units:
                raise TypeError(
                    'Sorry, argument "{}" expected to get value with unit {}, but got {}'.format(
                        arg_name,
                        expected.units.units,
                        given.units))

        if hasattr(given, 'magnitude'):
            magnitude = given.magnitude
        else:
            magnitude = given
        shape = len(str(magnitude).split())
        if not shape == expected.dimension:
            raise TypeError(
                'Argument {} expected to get value with dimension {}, but got dimension {}'.format(
                    arg_name,
                    expected.dimension,
                    shape))


class expects(object):

    """
    Decorator which determines expected arguments for the function
    and also check correctness of given arguments. If input arguments differ from
    expected ones, exception *TypeError* will be raised.

    For numeric arguments use *Numeric* class with 2 parameters: dimension of the array
    and units (optional). E.g. "Numeric (1)" means function expects one number or
    "Numeric (2, q.mm)" means function expects expression like [4,5]*q.mm

    Common use case looks like this:

    @expects (Camera, LinearMotor, pixelsize = Numeric(2, q.mm))
    def foo(camera, motor, pixelsize = None):
        pass
    """

    def __init__(self, *args, **kwargs):
        self.e_args = args
        self.e_keywords = kwargs

    def __call__(self, f):
        f_args = inspect.getargspec(f).args
        f_defaults = inspect.getargspec(f).defaults
        self.func = _Structure(
            f,
            self.e_args,
            f_args,
            f_defaults,
            self.e_keywords)
        return self.func


class Numeric(object):
    __name__ = "Numeric"

    def __init__(self, dimension, units=None):
        self.dimension = dimension
        self.units = units
