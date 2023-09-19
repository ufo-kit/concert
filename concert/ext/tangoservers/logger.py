"""
logger.py
---------
Implements a device server for logging in remote server
"""
import os
import platform
import logging
from tango import DebugIt, DevState
from tango.server import attribute, command, AttrWriteType, Device, DeviceMeta

class TangoRemoteLogger(Device, metaclass=DeviceMeta):
    
    path = attribute(
        label="Path",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_path",
        fset="set_path",
        doc="path where the log file needs to be written"
    )

    log_name = attribute(
        label="LogName",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_log_name",
        fset="set_log_name",
        doc="name of the log file"
    )

    _handler: logging.FileHandler
    _logger: logging.Logger
    
    def _init_handler(self) -> None:
        """Creates a file handler for the central logger object"""
        handler = logging.FileHandler(filename=self._path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(name)s: %(message)s")
        )
        self._handler = handler

    async def init_device(self) -> None:
        self.info_stream("%s init_device", self.__class__.__name__)
        await super().init_device()
        root = os.environ["HOME"]
        self._log_name = "experiment.log"
        self._path = f"{root}/{self._log_name}"
        self._logger = logging.getLogger(
                name=f"Logger@{platform.node()}")
        # Setting log level to debug to ensure that we capture all logs using
        # the file handler. logging.DEBUG has the lowest severity level after
        # the logging.NOTSET, which is the default level for the file handler.
        self._logger.setLevel(logging.DEBUG)
        self._init_handler()
        self._logger.addHandler(self._handler)
        self.set_state(DevState.STANDBY)
        self.info_stream(
                "logger in state: %s at path: %s",
                self.get_state(), self.get_path()
        )

    def get_path(self) -> str:
        return self._path

    def set_path(self, new_path: str) -> None:
        self._path = new_path
        # When remote directory walker descends to or ascends from a file path
        # we expect this setter to get a callback to update the path for the
        # logger. Subsequently, we switch the file handler for the logger after
        # removing and resetting the internal handler which we maintain.
        self.info_stream(
            "%s changing path",
            self.__class__.__name__
        )
        if self._logger.hasHandlers():
            self._logger.removeHandler(self._handler)
        self._init_handler()
        self._logger.addHandler(self._handler)
        self.set_state(DevState.STANDBY)
        self.info_stream(
            "%s in state: %s at path: %s",
            self.__class__.__name__,
            self.get_state(),
            self.get_path()
        )

    def get_log_name(self) -> None:
        return self._log_name

    def set_log_name(self, new_name: str) -> None:
        self._log_name = new_name

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def info_log(self, msg: str) -> None:
        self._logger.info(msg)

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def warning_log(self, msg: str) -> None:
        self._logger.warning(msg)
    
    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def debug_log(self, msg: str) -> None:
        self._logger.debug(msg)

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def error_log(self, msg: str) -> None:
        self._logger.error(msg)

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def critical_log(self, msg: str) -> None:
        self._logger.critical(msg)
    
    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def log(self, msg: str) -> None:
        self._logger.log(msg)


if __name__ == "__main__":
    pass

