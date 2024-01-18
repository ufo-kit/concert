"""
loghandler.py
-------------
Encapsulates the log handling utilities for the walker APIs.
"""
import logging
import asyncio
from typing import List, Protocol
from concert.typing import RemoteDirectoryWalkerTangoDevice
from concert.coroutines.base import get_event_loop

LOG = logging.getLogger(__name__)


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
    _tasks: List[asyncio.Task]
    _loop: asyncio.AbstractEventLoop

    def __init__(
            self,
            device: RemoteDirectoryWalkerTangoDevice,
            fmt: str = "[%(asctime)s] %(levelname)s: %(name)s: %(message)s") -> None:
        """
        Instantiates a remote handler for logging in remote host.

        :param device: device to send the log payload to write
        :type device: RemoteDirectoryWalkerTangoDevice
        :param fmt: format for logging
        :type fmt: str
        """
        super().__init__()
        self._device = device
        self.setFormatter(logging.Formatter(fmt))
        self._loop = get_event_loop()
        self._tasks = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Ensures conformance to logging.Handler api. In this method we create
        a coroutine task in the form of an asyncio.Future and add the same to
        collection of future tasks after adding a callback to handle the task
        completion.
        """
        if self._loop:
            self._tasks.append(
                asyncio.ensure_future(self._device.log([str(record.levelno),
                                                       self.format(record)]),
                                      loop=self._loop))
        else:
            raise RuntimeError("event loop unavailable")

    async def aflush(self) -> None:
        await asyncio.gather(*self._tasks)
        super().flush()

    async def aclose(self) -> None:
        await self.aflush()
        super().close()


if __name__ == "__main__":
    pass
