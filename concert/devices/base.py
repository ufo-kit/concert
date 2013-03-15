from logbook import Logger


log = Logger(__name__)


class UnknownStateError(Exception):
    """Any limit (hard or soft) exception."""
    def __init__(self, message):
        self._message = message

    def __str__(self):
        return repr(self._message)


class State(object):
    """State of a device.

    This is NOT a connection status, but a reflection of a physical device
    status. The implementation should follow this guideline.

    """
    ERROR = "error"
