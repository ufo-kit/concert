import os
from concert.coroutines import coroutine
from concert.storage import create_directory, write_tiff


class Result(object):
    """
    The object is callable and when called it becomes a coroutine which accepts
    items and stores them in a variable which allows the user to obtain the
    last stored item at any time point.
    """
    def __init__(self):
        self.result = None

    @coroutine
    def __call__(self):
        """
        __call__(self)

        When the object is called we store every item in an attribute which
        can be recovered later.
        """
        while True:
            self.result = yield


@coroutine
def null():
    """
    null()

    A black-hole.
    """
    while True:
        yield


@coroutine
def write_images(writer=write_tiff, prefix="image_{:>05}"):
    """
    write_images(writer, prefix="image_{:>05}")

    Write images on disk with specified *writer* and file name *prefix*.
    *writer* is a callable with the following nomenclature::

        writer(file_name_prefix, data)

    The file extension needs to be applied by a particular writer.
    """
    i = 0

    dir_name = os.path.dirname(prefix)
    if dir_name != "" and not os.path.exists(dir_name):
        create_directory(dir_name)

    while True:
        data = yield
        writer(prefix.format(i), data)
        i += 1
