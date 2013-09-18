"""Data transfers."""
from concert.base import coroutine


@coroutine
def multicast(*destinations):
    """Provide data to all *destinations*."""
    while True:
        item = yield
        for destination in destinations:
            destination.send(item)

