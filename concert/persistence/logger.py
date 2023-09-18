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
                        path: Optional[str], 
                        log_name: str = "experiment.log") -> None:
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
        self._log_name = log_name
        await self._device.write_attribute(attr_name="log_name",
                                           value=self._log_name)
        if path:
            self._path = f"{path}/{self._log_name}"
            await self._device.write_attribute(
                    attr_name="path", value=self._path)
        else:
            self._path = (await self._device["path"]).value
        await super().__ainit__()
    
    async def set_log_path(self, new_path: str) -> None:
        """
        Sets the path for logging along with log file name. Each time the
        walker descends to a given path in the remote file system, this method
        is called with the `current` directory of the walker.

        :param new_path: path for woriting the log file, usually the current
        directory of the remote walker
        :type new_path: str

        """
        self._path = new_path
        await self._device.write_attribute(
                attr_name="path", value=f"{self._path}/{self._log_name}")


if __name__ == "__main__":
    pass

