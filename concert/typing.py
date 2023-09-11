"""
typing.py
---------
Facilitates type annotations for concert
"""
from typing import Protocol, Any
import numpy
if numpy.__version__ >= "1.20":
    from numpy.typing import ArrayLike
else:
    from numpy import ndarray as ArrayLike
#####################################################################
# Error Types
class StorageError(Exception):
    """
    Exceptions related to logical issues with storage.
    """
    pass
#####################################################################


#####################################################################
# Abstract Device Types

class AbstractTangoDevice(Protocol):
    """
    Abstract Tango device which let's users to write arbitrary attribute as
    key value pairs.
    """

    async def write_attribute(self, attr_name: str, value: Any) -> None:
        """Lets the caller write a device attribute

        :param attr_name: attribute name
        :type attr_name: str
        :param value: attribute value
        :type value: str
        """

    def descend(self, name: str) -> None:
        """
        Creates and/or enters to a directory specified by name from current
        directory. Eventually, updates the current directory.
        :param name: directory name to create and/or enter
        :type name: str
        """

    def ascend(self) -> None:
        """
        Transition to one level up in the file system. Eventually, updates
        the current directory.
        """

    def exists(self, *paths: str) -> bool:
        """
        Asserts whether the specified paths exists in the file system.

        :param paths: a given number of file system paths
        :type paths: str
        :return: asserts whether specified path exists
        :rtype: bool
        """

    async def cancel(self) -> None:
        """
        TODO: Understand, what cancel does
        """

    async def reset_connection(self) -> None:
        """
        TODO: Understand, what reset_connection does
        """

    async def teardown(self) -> None:
        """
        TODO: Understand, what teardown does
        """
#####################################################################

