"""
*Processes* are software abstractions to control devices in a more
sophisticated way than just manipulating their parameters by hand. Each process
that is defined in this module provides one :meth:`run` method that is executed
asynchronously and returns whatever is appropriate for the process.
"""

from functools import wraps
from concert.base import Parameterizable
from concert.asynchronous import async


class Process(Parameterizable):

    """Base process."""

    def __init__(self, params):
        super(Process, self).__init__(params)

    @async
    def run(self):
        """run()

        Run the process. The result depends on the actual process.
        """
        raise NotImplementedError


def coroutine(func):
    """Start a generator automatically."""
    @wraps(func)
    def start(*args, **kwargs):
        """Starts the generator."""
        gen = func(*args, **kwargs)
        gen.next()
        return gen
    return start


def inject(generator, destination):
    """
    Let a *generator* produce a value and forward it to *destination*.
    """
    for item in generator:
        destination.send(item)


@coroutine
def multicast(*destinations):
    """
    multicast(*destinations)

    Provide data to all *destinations*.
    """
    while True:
        item = yield
        for destination in destinations:
            destination.send(item)
