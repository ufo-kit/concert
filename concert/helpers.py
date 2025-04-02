from __future__ import annotations

import asyncio
import copy
import os
import time
import inspect
import functools
import logging
import zmq
import concert.config
from dataclasses import dataclass, field
from typing import Any
from pint.errors import DimensionalityError
from concert.quantities import q

import numpy as np

LOG = logging.getLogger(__name__)


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
            values = dict(list(zip(values, values)))
        self.__dict__.update(values)


def memoize(func):
    """
    Memoize the result of *func*.

    Remember the result of *func* depending on its arguments. Note, that this
    requires that the function is free from any side effects, e.g. returns the
    same value given the same arguments.
    """
    memo = {}

    if inspect.iscoroutinefunction(func):
        async def wrapper(*args):
            if args in memo:
                return memo[args]

            result = await func(*args)
            memo[args] = result
            return result
    else:
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
            dim_units = expected.dimension * expected.units
            e_units = dim_units.to_base_units().units
            if not e_units == given.to_base_units().units:
                raise TypeError(
                    'Sorry, argument "{}" expected to get value with unit {}, but got {}'.format(
                        arg_name,
                        expected.units,
                        given.units))

        if hasattr(given, 'magnitude'):
            magnitude = given.magnitude
        else:
            magnitude = given
        if isinstance(magnitude, (float, int)):
            shape = 1
        else:
            shape = len(magnitude)
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

        from concert.helpers import Numeric

        @expects(Camera, LinearMotor, pixelsize = Numeric(2, q.mm))
        def foo(camera, motor, pixelsize = None):
            pass
    """

    def __init__(self, *args, **kwargs):
        self.e_args = args
        self.e_keywords = kwargs

    def __call__(self, f):
        f_args = inspect.getfullargspec(f).args
        f_defaults = inspect.getfullargspec(f).defaults
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


@dataclass(order=True)
class PrioItem:
    """To be used in combination with `queue.PriorityQueue`."""

    priority: int
    data: Any = field(compare=False)


def is_iterable(item):
    """Is *item* iterable or not."""
    try:
        iter(item)
        return True
    except Exception:
        return False


def arange(start, stop, step):
    """
    This function wraps numpy.arange but strips the units before and adds the unit later at the
    numpy.array.

    :param start:
    :type start: concert.quantities.q.Quantity
    :param stop:
    :type stop: concert.quantities.q.Quantity
    :param step:
    :type step: concert.quantities.q.Quantity
    :return:
    """
    try:
        unit = start.units
        start = start.to(unit).magnitude
        stop = stop.to(unit).magnitude
        step = step.to(unit).magnitude
    except AttributeError:
        raise Exception("start, stop and step need to be Quantities.")
    except DimensionalityError:
        raise Exception("start, stop and step units are not convertable into each other.")
    except Exception as e:
        raise e
    return np.arange(start=start, stop=stop, step=step) * unit


def linspace(start, stop, num, endpoint=True):
    """
    This function wraps numpy.linspace but strips the units before and adds the unit later at the
    numpy.array.

    :param start: First value
    :type start: concert.quantities.q.Quantity
    :param stop:
    :type stop: concert.quantities.q.Quantity
    :param num:
    :type num: int
    :param endpoint:
    :type endpoint: bool
    :return: numpy.array with the length *num* and entries equally distributed within *start* and
        *stop*.
    """
    try:
        unit = start.units
        start = start.to(unit).magnitude
        stop = stop.to(unit).magnitude
    except AttributeError:
        raise Exception("start and stop need to be Quantities.")
    except DimensionalityError:
        raise Exception("start and stop units are not convertable into each other.")
    except Exception as e:
        raise e
    return np.linspace(start, stop, num, endpoint=endpoint) * unit


async def get_state_from_awaitable(awaitable) -> str:
    if awaitable is None:
        return 'standby'
    task = asyncio.ensure_future(awaitable)
    if not task.done():
        return 'running'
    else:
        if task.cancelled():
            return 'cancelled'
        elif task.exception():
            return 'error'
        else:
            return 'standby'


class ImageWithMetadata(np.ndarray):
    """
    Subclass of numpy.ndarray with a metadata dictionary to hold images its metadata.
    """
    def __new__(cls, input_array, metadata: dict | None = None):
        obj = np.asarray(input_array).view(cls)
        obj.metadata.update(metadata)

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.metadata = copy.deepcopy(getattr(obj, "metadata", {}))
        if "conversion_applied" not in self.metadata:
            self.metadata["conversion_applied"] = False

    def _convert(self, forward):
        """
        Convert image and return a numpy ndarray, not ImageWithMetadata anymore in order to avoid
        nesting issues.
        """
        obj = self
        if self.metadata["conversion_applied"] ^ forward:
            for (operation, arg) in self.metadata.items():
                # If we have ordered dict the order of operations is preserved
                if operation == "mirror" and arg:
                    obj = np.fliplr(obj)
                if operation == "rotate":
                    obj = np.rot90(obj, k=arg if forward else -arg)

            obj.metadata["conversion_applied"] = forward

        return obj

    def convert(self):
        return self._convert(True)

    def convert_back(self):
        return self._convert(False)


class PerformanceTracker:

    """A stopwatch with the ability to report data throughput.

    :param summary: if True, output everything at the end, otherwise immediately on :meth:`.lap`
    call.
    :param loglevel: logging level
    """

    def __init__(self, summary=False, loglevel=concert.config.PERFDEBUG):
        self.start = 0 * q.s
        self.duration = 0 * q.s
        self.size = 0 * q.B
        self.loglevel = loglevel
        self.summary = [] if summary else None
        self.iteration = 0

    def __enter__(self):
        self.start = time.perf_counter() * q.s

        return self

    def lap(self, name=None, final=False):
        duration = time.perf_counter() * q.s - self.start

        if final:
            self.duration = duration

        if name is None:
            if final:
                name = 'final'
            else:
                name = str(self.iteration)
                self.iteration += 1

        output = f'{name} duration: {duration:.2f}'
        if self.size:
            size = self.size.to(q.mebibyte)
            output += f', size: {size:.2f}, throughput: {size / duration:.2f}'

        if self.summary is not None:
            self.summary.append(output)
        else:
            LOG.log(self.loglevel, output)

    def __exit__(self, exc_type, exc_value, traceback):
        self.lap(final=True)
        if self.summary is not None:
            for record in self.summary:
                LOG.log(self.loglevel, record)


@dataclass
class CommData:
    """Encapsulates communication metadata."""

    host: str
    port: int = None
    protocol: str = "tcp"
    socket_type: zmq.SocketType = zmq.PUSH
    sndhwm: int = 1000

    def __post_init__(self):
        if self.protocol not in ["tcp", "ipc"]:
            raise ValueError("protocol must be one of `tcp', `ipc'")

    @property
    def server_endpoint(self) -> str:
        if self.protocol == "ipc":
            return f"{self.protocol}://{self.host}"
        else:
            return f"{self.protocol}://*:{self.port}"

    @property
    def client_endpoint(self) -> str:
        if self.protocol == "ipc":
            return f"{self.protocol}://{self.host}"
        else:
            return f"{self.protocol}://{self.host}:{self.port}"

    def __eq__(self, other: CommData) -> bool:
        for key, value in self.__dict__.items():
            if other.__dict__[key] != value:
                return False
        return True

    def __ne__(self, other: CommData) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.host, self.port, self.protocol, self.socket_type, self.sndhwm))


def get_basename(filename):
    if filename.endswith(os.sep):
        filename = filename[:-1]

    return os.path.basename(filename)
