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
    Exception related to logical issues with storage.
    """


class SequenceReaderError(Exception):
    """
    Exception related to reading file sequence from disk storage
    """
    pass
#####################################################################


#####################################################################
# Abstract Tango Device Types
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


class RemoteDirectoryWalkerTangoDevice(AbstractTangoDevice, Protocol):
    """
    Abstract remote walker device type. While invoking these methods on a 
    Tango device server object we generally avoid using named arguments e.g., 
    descend(name="dir"). It was observed that this does not work well with
    Tango.
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

    async def write_sequence(self, path: str) -> None:
        """
        Asynchronously writes sequence of images in the provided path.

        :param path: path to write images to
        :type path: str
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

class RemoteLoggerTangoDevice(AbstractTangoDevice):
    """
    Abstract remote logger device type
    """
    async def debug(self, msg: str) -> None:
        """Handles debug logging for provided message"""

    async def info(self, msg: str) -> None:
        """Handles info logging for provided message"""

    async def warning(self, msg: str) -> None:
        """Handles warning logging for provided message"""
    
    async def error(self, msg: str) -> None:
        """Handles error logging for provided message"""

    async def critical(self, msg: str) -> None:
        """Handles critical logging for provided message"""
#####################################################################


if __name__ == "__main__":
    pass

