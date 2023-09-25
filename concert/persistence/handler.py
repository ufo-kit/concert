"""
handler.py
----------
Defines a remote logging handler which should facilitate logging at the remote
host at a specified location.
"""
import logging
import asyncio
from typing import List, Any
from concert.persistence.typing import RemoteDirectoryWalkerTangoDevice
from concert.coroutines.base import get_event_loop


class RemoteHandler(logging.Handler):

    _device: RemoteDirectoryWalkerTangoDevice
    _path: str
    _fname: str
    _futures: List[asyncio.Future]

    def __init__(self, 
                 device: RemoteDirectoryWalkerTangoDevice,
                 path: str,
                 fname: str = "experiment.log",
                 fmt: str = "[%(asctime)s] %(levelname)s: %(name)s: %(message)s"
                 ) -> None:
        """
        Instantiates a remote handler for logging in remote host.

        :param device: device to send the log payload to write
        :type device: RemoteDirectoryWalkerTangoDevice
        :param path: path to write the log file to
        :type path: str
        :param fname: log file name, defaults to `experiment.log`
        :type fname: str
        """
        super().__init__()
        self._device = device
        self._path = path
        self._fname = fname
        self._futures = []
        self.setFormatter(logging.Formatter(fmt))


    def _logging_task_cb(self, future: asyncio.Future) -> None:
        """
        Defines a callback function to remove the future from the collection
        of futures that the remote handler maintains.
        """
        if future in self._futures:
            self._futures.remove(future)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Ensures conformance to logging.Handler api.
        """
        loop: asyncio.AbstractEventLoop = get_event_loop()
        self._futures.append(
                asyncio.ensure_future(
                    await self._device.append_to_file(
                        [f"{self._path}/{self._fname}", self.format(record)]
                    ),
                    loop=loop
                ).add_done_callback(self._logging_task_cb)
        )

    async def close(self) -> None:
        """
        Waits until all the scheduled futures are done and then instructs the
        remote device to close the log file resource.
        """
        await asyncio.gather(self._futures)

#def clean_up_logger(logger: logging.Logger) -> logging.Logger:
#    """
#    Cleans up the remote handlers from the logger instance
#    """
#    assert logger is not None
#    if logger.hasHandlers():
#        for handler in logger.handlers:
#            if isinstance(handler, RemoteHandler):
#                logger.removeHandler(handler)
#    return logger


if __name__ == "__main__":
    pass

