from .base import coroutine


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
