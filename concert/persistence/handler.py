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
    _futures: List[asyncio.Future]
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
        self._futures = []

    def _logging_task_cb(self, future: asyncio.Future) -> None:
        """
        Defines a callback function to remove the future from the collection
        of futures that the remote handler maintains. This method is supposed
        to be called when the provided future has completed. At this point we
        no longer need to maintain the same in our collection of futures any
        more.
        """
        if future in self._futures:
            self._futures.remove(future)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Ensures conformance to logging.Handler api. In this method we create
        a coroutine task in the form of an asyncio.Future and add the same to
        collection of future tasks after adding a callback to handle the task
        completion.
        """
        self._futures.append(
            asyncio.ensure_future(
                self._device.log(self.format(record)), loop=self._loop
            ).add_done_callback(self._logging_task_cb)
        )

    def close(self) -> None:
        """
        Ensures conformance to logging.Handler api. Waits until all the
        scheduled futures are done and then instructs the remote device to
        close the log file resource. Latter has the to wait on the former,
        hence we can not use asyncio.gather on both futures.
        """
        asyncio.ensure_future(asyncio.gather(self._futures), loop=self._loop)
        asyncio.ensure_future(self._device.close_log_file(), loop=self._loop)


if __name__ == "__main__":
    pass

