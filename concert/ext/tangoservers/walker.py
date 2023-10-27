"""
walker.py
---------
Implements a device server for file system traversal at remote host.
"""
import io
import os
import re
from typing import Type, Optional, Awaitable, AsyncIterable
from tango import DebugIt, DevState, CmdArgType
from tango.server import attribute, command, AttrWriteType
from concert.helpers import PerformanceTracker
from concert.quantities import q
from concert import writers
from concert.storage import StorageError
from concert.typing import ArrayLike
from concert.storage import write_images, split_dsetformat, DirectoryWalker
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
  
    _writer: Type[writers.TiffWriter]
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
        self.info_stream(
            "%s has set current directory to: %s, with state: %s",
            self.__class__.__name__, self.get_current(), self.get_state()
        )

    def get_root(self) -> str:
        return self._root

    def set_root(self, path: str) -> None:
        self._root = path
        self._create_dir(directory=self._root)
        self.info_stream(
            "%s has set root directory to: %s, with state: %s",
            self.__class__.__name__, self.get_root(), self.get_state()
        )

    def set_writer_class(self, klass: str) -> None:
        # NOTE: With our current implementation we make this strong assertion.
        # We also take note that this approach is rather rigid and needs to be
        # reconsidered in future.
        assert klass == "TiffWriter"
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
    @command(
        dtype_in=str,
        dtype_out=bool,
        doc_in="path(s) to check if they exist"
    )
    def exists(self, *paths: str) -> bool:
        # TODO: Confirm on the datatype that this method is supposed to accept.
        # In normal python sense this is not important but since we implementing
        # a tango device server we need to specify the input datatype correctly.
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

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    async def write_sequence(self, path):
        walker = await DirectoryWalker(
            writer=self._writer,
            dsetname=self._dsetname,
            bytes_per_file=self._bytes_per_file,
            root=path
        )
        await self._process_stream(self.consume(walker))

    async def consume(self, walker: DirectoryWalker) -> None:
        with PerformanceTracker() as pt:
            total_bytes = await walker.write(self._receiver.subscribe())
            pt.size = total_bytes * q.B

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
 
