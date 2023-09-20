"""
logger.py
---------
Implements a device server for logging at a remote host.
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
        """
        Creates a file handler for the central logger object.

        NOTE: This method handles the deallocation of the previously allocated
        file handler pointing to a specific file resource in memory.
        """
        if self._handler is not None and isinstance(
                self._handler, logging.FileHandler):
            self._handler.close()
        handler = logging.FileHandler(
                filename=f"{self._path}/{self._log_name}",
                mode="a",
                encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(name)s: %(message)s")
        )
        self._handler = handler

    def _prepare_logger(self) -> None:
        """Prepares internal logger for upcoming logging tasks"""
        assert self._logger is not None
        # Setting log level to debug to ensure that we capture all logs using
        # the file handler. logging.DEBUG has the lowest severity level after
        # the logging.NOTSET, which is the default level for the file handler.
        self._logger.setLevel(logging.DEBUG)
        # Only viable when we change the logging path
        if self._logger.hasHandlers():
            self._logger.removeHandler(self._handler)
        # (Re)initialize handler and set the same to logger
        self._init_handler()
        self._logger.addHandler(self._handler)

    async def init_device(self) -> None:
        """
        Asynchronously initializes the device server.

        NOTE: In this implementation we don't instantiate the FileHandler
        object because the current file system context, namely the HOME
        directory is not the place where we want to persist the experiment
        logs. Since, instantiation of FileHandler automatically creates the
        log file this activity would be deferred until the directory walker
        walks to a reasonable location for logging.
        """
        await super().init_device()
        self._path = os.environ["HOME"]
        self._log_name = "experiment.log"
        self._logger = logging.getLogger(
                name=f"Logger@{platform.node()}")
        self._handler = None
        self.set_state(DevState.STANDBY)
        self.info_stream(
                "logger in state: %s at : %s",
                self.get_state(), f"{self.get_path()}/{self._log_name}"
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
        self._prepare_logger() 
        self.set_state(DevState.STANDBY)
        self.info_stream(
            "logger in state: %s ready to log at : %s",
            self.get_state(), f"{self.get_path()}/{self._log_name}"
        )
        
    def get_log_name(self) -> None:
        return self._log_name

    def set_log_name(self, new_name: str) -> None:
        self._log_name = new_name
        self.info_stream(
            "%s changing log file name",
            self.__class__.__name__
        )
        self._prepare_logger() 
        self.set_state(DevState.STANDBY)
        self.info_stream(
            "logger in state: %s ready to log at : %s",
            self.get_state(), f"{self.get_path()}/{self._log_name}"
        )

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def debug_log(self, msg: str) -> None:
        self._logger.debug(msg)

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
    def error_log(self, msg: str) -> None:
        self._logger.error(msg)

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    def critical_log(self, msg: str) -> None:
        self._logger.critical(msg)
    

if __name__ == "__main__":
    pass

