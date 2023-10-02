"""
walker.py
---------
Implements a device server for file system traversal at remote host.
"""
import io
import os
from typing import Type, Optional, Awaitable, AsyncIterable, List
import re
from tango import DebugIt, DevState
from tango.server import attribute, command, AttrWriteType
from concert.helpers import PerformanceTracker
from concert.quantities import q
from concert.persistence import writers
from concert.persistence.typing import StorageError, ArrayLike
from concert.persistence.storage import write_images, split_dsetformat
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
  
    _writer: Type[writers.ImageWriter]
    _log_file: Optional[io.TextIOWrapper]
        
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
        await super().init_device()
        self._root = os.environ["HOME"]
        self._current = self._root
        self._log_file = None
        self.set_state(DevState.STANDBY)
        self.info_stream(
                "%s in state: %s at directory: %s",
                self.__class__.__name__, self.get_state(), self.get_current()
        )
    
    def get_current(self) -> str:
        return self._current

    def set_current(self, path: str) -> None:
        self._current = path

    def get_root(self) -> str:
        return self._root

    def set_root(self, path: str) -> None:
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

    @DebugIt()
    @command(dtype_in=str)
    def descend(self, name: str) -> None:
        assert self._current is not None
        new_path: str = os.path.join(self._current, name)
        self._create_dir(directory=new_path)
        self._current = new_path
        self.info_stream(
            "%s walked into directory: %s, with state: %s",
            self.__class__.__name__, self.get_current(), self.get_state()
        )

    @DebugIt()
    @command()
    def ascend(self) -> None:
        if self._current == self._root:
            raise StorageError(f"Cannot break out of `{self._root}'.")
        self._current = os.path.dirname(self._current)
        self.info_stream(
            "%s walked into directory: %s, with state: %s",
            self.__class__.__name__, self.get_current(), self.get_state()
        )
        
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
    
    @DebugIt()
    @command(
        dtype_in=str,
        doc_in="payload to be appended to log file"
    )
    async def log(self, payload: str) -> None:
        try:
            if self._log_file and self._log_file.writable():
                self._log_file.write(payload)
                self._log_file.write("\n")
        except ValueError as err:
            self.error_stream(
                "%s failed to log %s [%s] - %s",
                self.__class.__name__,
                payload,
                str(err),
                self.get_state()
            )
        self.info_stream(
            "%s logged to file - %s",
            self.__class__.__name__,
            self.get_state()
        )

    @DebugIt()
    @command(
        dtype_in=str,
        doc_in="file path (typically for logging) to open as file resource"
    )
    async def open_log_file(self, file_path: str) -> None:
        if not self._log_file:
            self._log_file = open(file=file_path, mode="a", encoding="utf-8")
        self.info_stream(
            "%s opened %s for writing - %s",
            self.__class__.__name__,
            file_path,
            self.get_state()
        )
    
    @DebugIt()
    @command()
    async def close_log_file(self) -> None:
        if self._log_file and not self._log_file.closed:
            self._log_file.close()
            self._log_file = None
            self.info_stream(
                "%s closed log file - %s",
                self.__class__.__name__,
                self.get_state()
            )


if __name__ == "__main__":
    pass
 
