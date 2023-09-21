"""
logger.py
---------
Implements a custom handler to facilitate logging at the remote host.
"""
from typing import Optional
from concert.persistence.typing import RemoteLoggerTangoDevice
from concert.base import AsyncObject


class RemoteLogger(AsyncObject):
    """
    Defines a convenient utility class which works as the frontend of the
    remote log dispatcher tango device.
    """

    _device: RemoteLoggerTangoDevice
    _log_name: str
    _path: str

    async def __ainit__(self,
                        device: RemoteLoggerTangoDevice,
                        path: Optional[str] = None,
                        log_name: Optional[str] = None) -> None:
        """
        Instantiates LogDispatcher

        :param device:
        :type device: RemoteLogDispatcherTangoDevice
        :param path: path to write the log file. This path is initialized
        as the root of the directory walker and updated each time the walker
        descends to a given path
        :type path: str
        :param log_name: name of the log file defaults to `experiment.log`
        :type log_name: str
        """
        self._device = device
        if path:
            self._path = path
            await self._device.write_attribute(
                    attr_name="path", value=self._path)
        else:
            self._path = (await self._device["path"]).value
        if log_name:
            self._log_name = log_name
            await self._device.write_attribute(
                    attr_name="log_name", value=self._log_name)
        else:
            self._log_name = (await self._device["log_name"]).value
        await super().__ainit__()

    def get_logging_state(self) -> str:
        """Provides an accessor for the current file path for logging"""
        return f"{self._path}/{self._log_name}"
    
    async def set_logging_path(self, new_path: str) -> None:
        """
        Sets the path for logging along with log file name. Each time the
        walker descends to a given path in the remote file system, this method
        is called with the `current` directory of the walker.

        :param new_path: path for writing the log file, usually the current
        directory of the remote walker
        :type new_path: str

        """
        assert new_path is not None and new_path != ""
        self._path = new_path
        await self._device.write_attribute(attr_name="path", value=self._path)

    async def set_experiment_root(self, alternative: Optional[str] = None) -> None:
        """
        Sets the current path as the root directory of an experiment.
        
        NOTE: This utility function is introduced to control logging. Earlier
        a writer device server used to instantiate a DirectoryWalker for a
        specified path. As a result, DirectoryWalker never had a global view
        of the currently traversed state of the file system. 

        With our proposed approach, RemoteDirectoryWalker has a global view and
        its underlying device server facilitates writing of acquisition data.
        Hence, we need some way to let it know, if some directory has a special
        significance e.g., with our current approach we tend to avoid logging
        at the root of an experiment. Instead, we prefer logging for individual
        acquisition. With optional toggle of the logging utility this method
        designates a given directory as the root of our experiment to let the
        system know, where not to create log files.

        :param alternative: an alternative path to be set as experiment root.
        Might be useful in rare occasions. Once should consider the currently
        traversed file system state before using this parameter.
        :type alternative: Optional[str]
        """
        exclusion: str = alternative if alternative is not None else self._path
        await self._device.write_attribute(attr_name="exclusion",
                                           value=exclusion)
   
    async def set_log_name(self, new_name: str) -> None:
        """
        Sets the log file name
        :param new_name: log file name
        :type new_name: str
        """
        self._log_name = new_name
        await self._device.write_attribute(attr_name="log_name",
                                           value=self._log_name)

    async def debug(self, msg: str) -> None:
        await self._device.debug_log(msg)

    async def info(self, msg: str) -> None:
        await self._device.info_log(msg)

    async def warning(self, msg: str) -> None:
        await self._device.warning_log(msg)

    async def error(self, msg: str) -> None:
        await self._device.error_log(msg)

    async def critical(self, msg: str) -> None:
        await self._device.critical_log(msg)


if __name__ == "__main__":
    pass

