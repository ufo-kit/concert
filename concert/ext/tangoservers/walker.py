"""
walker.py
---------
Implements device server for remote directory walker
"""
import os
import logging
from typing import Type, Optional, Awaitable, AsyncIterable
import re
from tango import DebugIt, DevState
from tango.server import attribute, command, AttrWriteType
from concert.helpers import PerformanceTracker
from concert.quantities import q
from concert.typing import StorageError, ArrayLike
from concert import writers
from concert.storage import write_images, split_dsetformat
from concert.ext.tangoservers.base import TangoRemoteProcessing


class TangoRemoteWalker(TangoRemoteProcessing):
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
        label="WriterClass",
        dtype=str,
        access=AttrWriteType.WRITE,
        fset="set_writer_class"
    )

    dsetname = attribute(
        label="Dsetname",
        dtype=str,
        access=AttrWriteType.WRITE,
        fset="set_dsetname"
    )

    bytes_per_file = attribute(
        label="BytesPerFile",
        dtype=int,
        access=AttrWriteType.WRITE,
        fset="set_bytes_per_file",
    )

    start_index = attribute(
        label="StartIndex",
        dtype=int,
        access=AttrWriteType.WRITE,
        fset="set_start_index"
    )

    logger_class = attribute(
        label="LoggerClass",
        dtype=str,
        access=AttrWriteType.WRITE,
        fset="set_logger_class"
    )

    log_name = attribute(
        label="LogName",
        dtype=str,
        access=AttrWriteType.WRITE,
        fset="set_log_name"
    )

    _writer: Type[writers.ImageWriter]
    _logger: Optional[logging.Logger]
    _log_handler: Optional[logging.FileHandler]
    
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

    async def init_device(self) -> None:
        """
        Initializes the remote walker tango device. Sets the remote server's
        home directory as root as well as the current directory.
        """
        self.info_stream("%s init_device", self.__class__.__name__)
        await super().init_device()
        self._root = os.environ["HOME"]
        self._current = self._root
        self.set_state(DevState.STANDBY)
        self.info_stream(
                "%s in state: %s in directory: %s",
                self.__class__.__name__, self.get_state(), self.get_root()
        )
    
    def get_current(self) -> str:
        return self._current

    def set_current(self, path: str) -> None:
        self._current = path

    def get_root(self) -> str:
        return self._root

    def set_root(self, path: Optional[str] = None) -> None:
        if path:
            self._root = path
            self._current = self._root
            self._create_dir(directory=self._root)

    def set_writer_class(self, klass: str) -> None:
        self._writer = getattr(writers, klass)

    def set_dsetname(self, dsetname: str) -> None:
        self._dsetname = dsetname

    def set_bytes_per_file(self, val: int) -> None:
        self._bytes_per_file = val

    def set_start_index(self, idx: int) -> None:
        self._start_index = idx

    def set_log_name(self, log_name: str) -> None:
        self._log_name = log_name

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
            assert self._root is not None and self._log_name is not None
            self._log_handler = logging.FileHandler(
                    os.path.join(self._root, self.bytes_per_file_log_name))
    
    @DebugIt()
    @command(dtype_in=str)
    def descend(self, name: str) -> None:
        assert self._current is not None
        new_path: str = os.path.join(self._current, name)
        self._create_dir(directory=new_path)
        self._current = new_path

    @DebugIt()
    @command()
    def ascend(self) -> None:
        if self._current == self._root:
            raise StorageError(f"Cannot break out of `{self._root}'.")
        self._current = os.path.dirname(self._current)

    @DebugIt()
    @command()
    def exists(self, *paths: str) -> bool:
        return os.path.exists(os.path.join(self._current, *paths))

    def _dset_exists(self, dsetname: str) -> bool:
        """
        Checks if the dataset exists at the current lavel of file system
        """
        if not re.match('.*{.*}.*', dsetname):
            raise ValueError(f"dataset name {dsetname} has wrong format")
        for f_name in os.listdir(self._current):
            if f_name.startswith(split_dsetformat(dsetname)):
                return True
        return False

    @DebugIt()
    @command()
    def create_writer(self, 
                      producer: AsyncIterable[ArrayLike],
                      dsetname: Optional[str] = None) -> Awaitable:
        if dsetname:
            dsn = dsetname
        else:
            dsn = self._dsetname
        if self._dset_exists(dsetname=dsn):
            dset_prefix = split_dsetformat(dsn)
            dset_path = os.path.join(self._current, dset_prefix)
            raise StorageError(f"{dset_path} is not empty")
        prefix = os.path.join(self._current, dsn)
        return write_images(
            producer,
            self._writer,
            prefix,
            self._start_index,
            self._bytes_per_file
        )
    
    async def _consume(self, produced: AsyncIterable[ArrayLike]) -> None:
        """
        Defines a wrapper corotuine which would be asynchronously handled
        by stream processing utility.

        :param producer: writable payload received at receiver endpoint
        :type producer: AsyncIterable[ArrayLike]
        """
        with PerformanceTracker() as prt:
            bytes_written: int = await self.create_writer(produced)
            prt.size = bytes_written * q.B

    @DebugIt()
    @command()
    async def write_sequence(self, path: str) -> None:
        # NOTE: Current writer device server, upon receiving a command to write
        # to a specified path, instantiates a local directory walker with the 
        # path. That essentially means that it sets the said local walker's root
        # and current directory according to the provided path.
        #
        # However, with this revised approach of combined walker and writer we
        # consider that the root may not need to be changed. Because unlike
        # before this device server now needs to maintain a global view of the
        # file system that it has traversed so far which starts at root.
        # Hence, we consider to set only the current directory according to
        # path while ensuring that the directory exists before writing to it.
        # We deal with this by directly descending to `path`.
        assert path is not None and path != ""
        self.descend(path)
        # Q: Apparently the subscribe method of the ZMQReceiver can yield
        # both metadata and image array if we make use of return_metadata
        # option. Hence, we need some consideration. For now we stick to
        # the approach taken by current writer device server.
        produced: AsyncIterable[ArrayLike] = await self._receiver.subscribe()
        await self._process_stream(self._consume(produced))       

if __name__ == "__main__":
    pass
 
