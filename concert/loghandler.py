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
class NoOpLoggingHandler:
    """Defines placeholder handler closer which can be used for dummy
    implementation."""

    async def aflush(self) -> None:
        """Defines a no-op asynchronous flush method"""
        ...

    async def aclose(self) -> None:
        """Defines a no-op asynchronous close method"""
        ...


class RemoteLoggingHandler(logging.Handler):
    """
    Facilitates logging to file at a remote server in the network by extending
    the builtin logging.Handler object and encapsulating a Tango device server
    running in the background.
    """

    _device: RemoteDirectoryWalkerTangoDevice
    _log_path: str
    _tasks: List[asyncio.Task]
    _loop: asyncio.AbstractEventLoop

    def __init__(
            self,
            device: RemoteDirectoryWalkerTangoDevice,
            log_path: str,
            fmt: str = "[%(asctime)s] %(levelname)s: %(name)s: %(message)s") -> None:
        """
        Instantiates a remote handler for logging in remote host.

        :param device: device to send the log payload to write
        :type device: RemoteDirectoryWalkerTangoDevice
        :param log_path: log path as unique identifier for the logger object
        :type log_path: str
        :param fmt: format for logging
        :type fmt: str
        """
        super().__init__()
        self._device = device
        self._log_path = log_path
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
                asyncio.ensure_future(self._device.log([self._log_path, str(record.levelno),
                                                        self.format(record)]), loop=self._loop))
        else:
            raise RuntimeError("event loop unavailable")

    async def aflush(self) -> None:
        await asyncio.gather(*self._tasks)
        super().flush()

    async def aclose(self) -> None:
        await self.aflush()
        await self._device.deregister_logger(self._log_path)
        super().close()


if __name__ == "__main__":
    pass
