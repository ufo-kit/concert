# pylint: disable=unnecessary-ellipsis

"""
typing.py
---------
Facilitates type annotations for concert
"""
from typing import Protocol, Any, NewType, Sequence
import numpy

# Defines ArrayLike as a new type
# NOTE: We take this approach because NumPy>=1.20 offers ArrayLike as a
# concrete type. At this point Tango has some discrepancy when it comes to
# NumPy versions. In future this can(should) be replaced with
# from numpy.typing import ArrayLike
ArrayLike = NewType("ArrayLike", numpy.ndarray)


#####################################################################
# Abstract Tango Device Types
class AbstractStreamHandler(Protocol):
    """
    Encapsulates specific stream handling utilities provided by the
    TangoRemoteProcessing base class to control the incoming data streams over
    the ZmqReceiver socket endpoint.
    """

    async def cancel(self) -> None:
        """
        Cancels any asynchronous task, maintained by the stream handler
        """
        ...

    async def reset_connection(self) -> None:
        """
        Stops receiving data from the current ZmqReceiver socket connection and
        reconnects the socket.
        """
        ...

    async def teardown(self) -> None:
        """
        Stops receiving data from the current ZmqReceiver socket connection
        without possibility of recovery.
        """
        ...


class AbstractTangoDevice(Protocol):
    """
    Abstract Tango device which let's users to write arbitrary attribute as
    key value pairs.
    """

    def get_attribute_list(self) -> Sequence[str]:
        """
        Returns a list of attributes from the device server.
        """
        ...

    async def write_attribute(self, attr_name: str, value: Any) -> None:
        """Lets the caller write a device attribute

        :param attr_name: attribute name
        :type attr_name: str
        :param value: attribute value
        :type value: str
        """
        ...

    async def __getitem__(self, key: Any) -> Any:
        """Exposes a dictionary-like interface for abstract tango device"""
        ...


class RemoteDirectoryWalkerTangoDevice(
        AbstractTangoDevice, AbstractStreamHandler, Protocol):
    """
    Abstract remote walker device type. While invoking these methods on a
    Tango device server object we generally avoid using named arguments e.g.,
    descend(name="dir"). It was observed that this does not work well with
    Tango.
    """

    async def descend(self, name: str) -> None:
        """
        Creates and/or enters to a directory specified by name from current
        directory. Eventually, updates the current directory.
        :param name: directory name to create and/or enter
        :type name: str
        """
        ...

    async def ascend(self) -> None:
        """
        Transition to one level up in the file system. Eventually, updates
        the current directory.
        """
        ...

    async def exists(self, *paths: str) -> bool:
        """
        Asserts whether the specified paths exists in the file system.

        :param paths: a given number of file system paths
        :type paths: str
        :return: asserts whether specified path exists
        :rtype: bool
        """
        ...

    async def write_sequence(self, path: str) -> None:
        """
        Asynchronously writes sequence of images in the provided path.

        :param path: path to write images to
        :type path: str
        """

    async def open_log_file(self, file_path: str) -> None:
        """
        Opens a log file for writing logs asynchronously.
        :param file_path: absolute path to the log file
        :type file_path: str
        """
        ...

    async def close_log_file(self) -> None:
        """Closes the log file if its open"""
        ...

    async def log(self, payload: str) -> None:
        """
        Writes log to the log file.
        :param payload: arbitrary log payload as a string
        :type payload: str
        """
        ...
#####################################################################


if __name__ == "__main__":
    pass
