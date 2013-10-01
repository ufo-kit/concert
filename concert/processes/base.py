"""
*Processes* are software abstractions to control devices in a more
sophisticated way than just manipulating their parameters by hand. Each process
that is defined in this module provides one :meth:`run` method that is executed
asynchronously and returns whatever is appropriate for the process.
"""

from functools import wraps
from concert.base import Parameterizable
from concert.helpers import async


class Process(Parameterizable):

    """Base process."""

    def __init__(self, params=None):
        super(Process, self).__init__(params)

    @async
    def run(self):
        """run()

        Run the process. The result depends on the actual process.
        """
        raise NotImplementedError
