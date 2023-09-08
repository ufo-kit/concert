"""
walker.py
---------
Implements device server for remote directory walker
"""

import os
from logging import Logger, FileHandler
from typing import Type, Optional, Awaitable, AsyncIterable
import re
from tango import DebugIt, InfoIt
from tango.server import attribute, command, AttrWriteType
from tango.server import Device, DeviceMeta
import numpy
if numpy.__version__ >= "1.20":
    from numpy.typing import ArrayLike
else:
    from numpy import ndarray as ArrayLike
from concert.typing import StorageError
from concert import writers
from concert.storage import write_images, split_dsetformat


class TangoRemoteWalker(Device, metaclass=DeviceMeta):
    """Tango device for filesystem walker in a remote server"""
    
    current = attribute(
        label="Current",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_current",
        fset="set_current",
        doc="current directory of context for remote file system walker"
    )

    root = attribute(
        label="Root",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_root",
        fset="set_root",
        doc="root directory for the remote file system walker"
    )

    writer_class = attribute(
        label="WriterClass"
        drype=str,
        access=AttrWriteType.READ_WRITE,
        fset="set_writer_class"
    )

    dsetname = attribute(
        label="Dsetname",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fset="set_dsetname"
    )

    bytes_per_file = attribute(
        label="BytesPerFile",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fset="set_bytes_per_file",
    )

    start_index = attribute(
        label="StartIndex",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fset="set_start_index"
    )

    logger_class = attribute(
        label="LoggerClass",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fset="set_logger_class"
    )

    log_name = attribute(
        label="LogName",
        drype=str,
        access=AttrWriteType.READ_WRITE,
        fset="set_log_name"
    )

    _writer: Type[writers.TiffWriter]
    _logger: Optional[logging.Logger]
    _log_handler: Optional[FileHandler]
    
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
    
    def get_current(self) -> str:
        return self.__current

    def set_current(self, path: str) -> None:
        self.__current = path

    def get_root(self) -> str:
        return self.__root

    def set_root(self, path: Optional[str] = None) -> None:
        if not path:
            self.__root = os.path.abspath(os.getcwd())
        else:
            self.__root = path
        self.__current = self.__root
        self._create_dir(directory=self.__root)

    def set_writer_class(self, klass: str) -> None:
        self._writer = getattr(writers, klass)

    def set_dsetname(self, dsetname: str) -> None:
        self.__dsetname = dsetname

    def set_bytes_per_file(self, val: int) -> None:
        self.__bytes_per_file = val

    def set_start_index(self, idx: int) -> None:
        self.__start_index = idx

    def set_log_name(self, log_name: str) -> None:
        self.__log_name = log_name

    def set_logger_class(self, klass: Optional[str] = None) -> None:
        # TODO: Understand the drawback of this approach. We are assuming
        # that the optional value would be 'Logger' and therefore we can use the
        # getattr function on the logging module. But what if that's not the
        # case. We have a similar situation with the set_writer_class method
        # as well. Need to verify the behavior of the tango attributes in this
        # regard. How strict it is in terms of defined dtype because options
        # for dtypes in the documentation are limited.
        # https://pytango.readthedocs.io/en/stable/data_types.html#pytango-data-types
        if klass:
            self._logger = getattr(logging, klass)
            assert self.__root is not None and self.__log_name is not None
            self._log_handler = FileHandler(
                    os.path.join(self.__root, self.__log_name))
    
    @DebugIt()
    @command(dtype_in=str)
    def descend(self, name: str) -> None:
        assert self.__current is not None
        new_path: str = os.path.join(self.__current, name)
        self._create_dir(directory=new_path)
        self.__current = new_path

    @DebugIt()
    @command()
    def ascend(self) -> None:
        if self.__current == self.__root:
            raise StorageError(f"Cannot break out of `{self.__root}'.")
        self.__current = os.path.dirname(self.__current)

    @DebugIt()
    @command()
    def exists(self, *paths: str) -> bool:
        return os.path.exists(os.path.join(self.__current, *paths))

    def _dset_exists(self, dsetname: str) -> bool:
        """
        Checks if the dataset exists at the current lavel of file system
        """
        if not re.match('.*{.*}.*', dsetname):
            raise ValueError(f"dataset name {dsetname} has wrong format")
        for f_name in os.listdir(self.__current):
            if f_name.startswith(split_dsetformat(dsetname)):
                return True
        return False

    @DebugIt()
    @command()
    def create_writer(self, 
                      producer: AsyncIterable[ArrayLike], 
                      dsetname: Optional[str] = None) -> Awaitable:
        dsn = self.__dsetname or dsetname
        if self._dset_exists(dsetname=dsn):
            dset_prefix = split_dsetformat(dsn)
            dset_path = os.path.join(self.__current, dset_prefix)
            raise StorageError(f"{dset_path} is not empty")
        prefix = os.path.join(self.__current, dsn)
        return write_images(
            producer,
            self._writer,
            prefix,
            self.__start_index,
            self.__bytes_per_file
        )


if __name__ == "__main__":
    pass
 
