"""Data transfers."""
from concert.base import coroutine


@coroutine
def multicast(*destinations):
    """Provide data to all *destinations*."""
    while True:
        item = yield
        for destination in destinations:
            destination.send(item)


@coroutine
def write_images(writer, prefix="radio_"):
    """
    Write images on disk with specified *writer* and file name *prefix*.
    *writer* is a callable with the following nomenclature::

        writer(file_name_prefix, data)

    The file extension needs to be applied by a particular writer.
    """
    i = 0

    while True:
        data = yield
        file_name_prefix = prefix.join("{:>06}".format(i))
        writer(file_name_prefix, data)
        i += 1
