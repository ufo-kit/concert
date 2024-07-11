"""
loghandler.py
-------------
Encapsulates the log handling utilities for the walker APIs.
"""
import asyncio
import hashlib
import logging
import uuid
from typing import List, Protocol, Callable, Any
from concert.typing import RemoteDirectoryWalkerTangoDevice
from concert.coroutines.base import get_event_loop

#############
# Protocols #
##############################################################################
class AsyncLoggingHandlerCloser(Protocol):
    """Abstarct logging handler, which can be closed asynchronously"""

    async def aflush(self) -> None:
        """Defines an asynchronous flushing routine for log entries"""
        ...

    async def aclose(self) -> None:
        """Defines an asynchronous closing routine for logging handler"""
        ...
##############################################################################

##############################################################################
# Convenient routine to create a unique identifier for an object. This identifier
# distinguishes the logger and in turn handler objects for remote walker device server.
uid_for: Callable[[object], str] = lambda obj: uuid.UUID(
        hashlib.md5(obj.__str__().encode("UTF-8")).hexdigest()).__str__()
##############################################################################

class NoOpLoggingHandler:
    """Defines placeholder handler closer which can be used for dummy
    implementation."""

    async def aflush(self) -> None:
        """Defines a no-op asynchronous flush method"""
        ...

    async def aclose(self) -> None:
        """Defines a no-op asynchronous close method"""
        ...


class LoggingHandler(logging.FileHandler):
    """Facilitates the logging utility for local directory traversal by
    extending builtin logging.Handler object."""

    def __init__(
            self,
            *args,
            fmt: str = "[%(asctime)s] %(levelname)s: %(name)s: %(message)s") -> None:
        super().__init__(*args)
        self.setFormatter(logging.Formatter(fmt))

    async def aflush(self) -> None:
        super().flush()

    async def aclose(self) -> None:
        super().close()


class RemoteLoggingHandler(logging.Handler):
    """
    Facilitates logging to file at a remote server in the network by extending
    the builtin logging.Handler object and encapsulating a Tango device server
    running in the background.
    """

    _device: RemoteDirectoryWalkerTangoDevice
    _owner_id: str
    _tasks: List[asyncio.Task]
    _loop: asyncio.AbstractEventLoop

    def __init__(
            self,
            device: RemoteDirectoryWalkerTangoDevice,
            owner_id: str,
            fmt: str = "[%(asctime)s] %(levelname)s: %(name)s: %(message)s") -> None:
        """
        Instantiates a remote handler for logging in remote host.

        :param device: device to send the log payload to write
        :type device: RemoteDirectoryWalkerTangoDevice
        :param owner_id: unique identifier for the object which wants to log something
        :type owner_id: str
        :param fmt: format for logging
        :type fmt: str
        """
        super().__init__()
        self._device = device
        self._owner_id = owner_id
        self.setFormatter(logging.Formatter(fmt))
        self._loop = get_event_loop()
        self._tasks = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Ensures conformance to logging.Handler api. In this method we create
        a coroutine task in the form of an asyncio.Future and add the same to
        collection of future tasks after adding a callback to handle the task
        completion. Owner_id is used by the device server to identify the logger
        object, which should deal with logging.
        """
        if self._loop:
            self._tasks.append(
                asyncio.ensure_future(self._device.log([self._owner_id, str(record.levelno),
                                                        self.format(record)]), loop=self._loop))
        else:
            raise RuntimeError("event loop unavailable")

    async def aflush(self) -> None:
        await asyncio.gather(*self._tasks)
        super().flush()

    async def aclose(self) -> None:
        await self.aflush()
        await self._device.deregister_logger_with(self._owner_id)
        super().close()


if __name__ == "__main__":
    pass
