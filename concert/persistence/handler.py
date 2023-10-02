"""
handler.py
----------
Defines a remote logging handler which should facilitate logging at the remote
host at a specified location.
"""
import logging
import asyncio
from typing import List
from concert.persistence.typing import RemoteDirectoryWalkerTangoDevice
from concert.coroutines.base import get_event_loop


class RemoteHandler(logging.Handler):
    """
    Facilitates a custom handler for remote logging by extending the
    logger.Handler api and encapsulating a Tango device.
    """

    _device: RemoteDirectoryWalkerTangoDevice
    _tasks: List[asyncio.Task]
    _loop: asyncio.AbstractEventLoop

    def __init__(
            self,
            device: RemoteDirectoryWalkerTangoDevice,
            fmt: str = "[%(asctime)s] %(levelname)s: %(name)s: %(message)s"
            ) -> None:
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
                self._loop.create_task(self._device.log(self.format(record)))
            )
        else:
            raise RuntimeError("event loop unavailable")

    async def aclose(self) -> None:
        """
        Defines an asynchronous routine to execute the collected tasks and
        clean up the resources.
        """
        await asyncio.gather(*self._tasks)
        await self._device.close_log_file()
        super().close()


if __name__ == "__main__":
    pass

