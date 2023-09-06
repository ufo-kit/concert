"""Implements device server for remote directory walker"""

import os
from tango import DebugIt, InfoIt
from tango.server import attribute, command, AttrWriteType
from tango.server import Device, DeviceMeta
from .base import TangoRemoteProcessing
from concert.typing import StorageError


class RemoteWalker(Device, metaclass=DeviceMeta):
    """Tango device for filesystem walker in a remote server"""
    
    _current: str

    root = attribute(
        label="Root",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_root",
        fset="set_root",
        doc="root directory for the remote file system walker"
    )
    
    @staticmethod
    def _create_dir(directory: str, mode: int = 0o0750) -> None:
        """
        Creates the given directory tree if not existing

        :param directory: directory tree to be created
        :type directory: str
        :param mode: owner right for the directory
        :type rights: int
        """
        if not os.path.exists(directory):
            os.makedirs(name=directory, mode=mode)


    def get_root(self) -> str:
        return self.__root

    def set_root(self, root: str) -> None:
        self.__root = root
        self._current = self.__root
        self._create_dir(directory=root)

    @DebugIt()
    @command(dtype_in=str)
    def descend(self, path: str) -> None:
        new_path: str = os.path.join(path=directory)
        self._create_dir(directory=new_path)
        self._current = new_path


    @DebugIt()
    @command()
    def ascend(self) -> None:
        if self._current == self.__root:
            raise StorageError(f"Cannot break out of `{self.__root}'.")
        self._current = os.path.dirname(self._current)

    
    @DebugIt()
    @command()
    def exists(self, *paths) -> bool:
        return os.path.exists(os.path.join(self._current, *paths))



if __name__ == "__main__":
    pass
    
