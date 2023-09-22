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
                 fname: str = "experiment.log"
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
        async self._device.append_to_file(
                [f"{self_path}/{self._fname}", payload])

    def emit(self, record: logging.LogRecord) -> None:
        asyncio.run(self.send(str(record)))

if __name__ == "__main__":
    pass

