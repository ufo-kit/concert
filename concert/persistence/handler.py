"""
handler.py
----------
Defines a remote logging handler which should facilitate logging at the remote
host at a specified location.
"""
import logging
from concert.persistence.typing import RemoteDirectoryWalkerTangoDevice
import asyncio


class RemoteHandler(logging.Handler):

    _device: RemoteDirectoryWalkerTangoDevice
    _path: str
    _fname: str

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
        self.setFormatter(logging.Formatter(fmt))

    async def send(self, payload: str) -> None:
        """
        Facilitates writing of the log record using device server

        :param payload: content to log
        :type payload: str
        """
        await self._device.append_to_file(
                [f"{self._path}/{self._fname}", payload])

    def emit(self, record: logging.LogRecord) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        asyncio.ensure_future(self.send(str(record)), loop=loop)

def clean_up_logger(logger: logging.Logger) -> logging.Logger:
    """
    Cleans up the remote handlers from the logger instance
    """
    assert logger is not None
    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, RemoteHandler):
                logger.removeHandler(handler)
    return logger


if __name__ == "__main__":
    pass

