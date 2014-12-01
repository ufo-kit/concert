import time
import inspect
import functools


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


def measure(func=None, return_result=False):
    """
    Measure and print execution time of *func*.

    If *return_result* is True, the decorated function returns a tuple
    consisting of the original return value and the measured time in seconds.
    """

    if func is not None:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            if return_result:
                from concert.quantities import q
                return (result, elapsed * q.s)
            else:
                print("`{}' took {}s".format(func.__name__, elapsed))
                return result
        return wrapper
    else:
        return functools.partial(measure, return_result=return_result)


class _Structure(object):

    def __init__(self, func, e_args, f_args, f_defaults, e_keywords):
        self.func = func
        self.e_args = e_args
        self.f_args = f_args
        self.f_defaults = f_defaults
        self.outputs = e_keywords['output']
        self.e_keywords = e_keywords
        self._isfunction = True

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
        import inspect
        if expected is not None:
            if isinstance(expected, Numeric):
                self._check_numeric(arg_name, expected, given)

            elif inspect.isclass(expected):
                if issubclass(expected, Device) and not isinstance(given, expected):
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

    For numeric arguments use *Numeric* class with 2 parameters: dimension of
    the array and units (optional). E.g. "Numeric (1)" means function expects
    one number or "Numeric (2, q.mm)" means function expects expression like
    [4,5]*q.mm

    Common use case looks like this::

        @expects(Camera, LinearMotor, pixelsize = Numeric(2, q.mm))
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
        functools.update_wrapper(self.func, f)
        return self.func


class Numeric(object):
    __name__ = "Numeric"

    def __init__(self, dimension, units=None):
        self.dimension = dimension
        self.units = units


class Region(object):

    """A Region holds a :class:`~concert.base.Parameter` and *values* which are the x-values of a
    scan. You can create the values e.g. by numpy's *linspace* function::

        import numpy as np
        # minimum=0, maximum=10, intervals=100
        values = np.linspace(0, 10, 100) * q.mm
    """

    def __init__(self, parameter, values):
        self.parameter = parameter
        self.values = values

    def __iter__(self):
        """Return region's iterator over its *values*."""
        return iter(self.values)

    def __repr__(self):
        return 'Region({})'.format(str(self))

    def __str__(self):
        return '{}: {}'.format(self.parameter.name, self.values)
