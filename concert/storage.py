"""Storage implementations."""
from __future__ import annotations
import asyncio
import os
import logging
import re
from typing import Optional, AsyncIterable, Awaitable, Type, Iterable, Set
import tifffile
from concert.base import AsyncObject
from concert.coroutines.base import background
from concert.writers import TiffWriter
from concert.typing import RemoteDirectoryWalkerTangoDevice
from concert.typing import ArrayLike
from concert.loghandler import AsyncLoggingHandlerCloser, NoOpLoggingHandler
from concert.loghandler import LoggingHandler, RemoteLoggingHandler


LOG = logging.getLogger(__name__)


def split_dsetformat(dsetname):
    """
    Strip *dsetname* off the formatting part wihch leaves us with the data
    set name.

    :param dsetname: dataset name
    :type dsetname: str
    """
    return dsetname.split('{')[0]


def read_image(filename):
    """
    Read image from file with *filename*. The file type is detected
    automatically.
    """
    for ext, reader in list(READERS.items()):
        if filename.lower().endswith(ext):
            return reader(filename)

    raise ValueError("Unsupported file type")


def read_tiff(file_name):
    """Read tiff file from disk by :py:mod:`tifffile` module."""
    with tifffile.TiffFile(file_name) as f:
        return f.asarray(out='memmap')


READERS = {".tif": read_tiff,
           ".tiff": read_tiff}

try:
    import fabio

    def read_edf_via_fabio(filename):
        edf = fabio.edfimage.edfimage()
        edf.read(filename)
        return edf.data

    for ext in ('.edf', '.edf.gz'):
        READERS[ext] = read_edf_via_fabio
except ImportError:
    pass


def write_tiff(file_name, data):
    """
    The default TIFF writer which uses :py:mod:`tifffile` module.
    Return the written file name.
    """
    tifffile.imsave(file_name, data)

    return file_name


def write_libtiff(file_name, data):
    """Write a TIFF file using pylibtiff. Return the written file name."""
    from libtiff import TIFF

    tiff_file = TIFF.open(file_name, "w")
    try:
        tiff_file.write_image(data)
    finally:
        tiff_file.close()

    return file_name


def create_directory(directory, rights="750"):
    """Create *directory* and all paths along the way if necessary. *rights* are a string
    representing a combination for user, group, others.
    """
    if not os.path.exists(directory):
        LOG.debug("Creating directory {}".format(directory))
        os.makedirs(directory, int(rights, base=8))


async def write_images(producer: AsyncIterable[ArrayLike],
                       writer: Type[TiffWriter] = TiffWriter,
                       prefix: str = "image_{:>05}.tif",
                       start_index: int = 0,
                       bytes_per_file: int = 0,
                       rights: str = "750") -> int:
    """
    write_images(pqueue, writer=TiffWriter, prefix="image_{:>05}.tif",
        start_index=0, bytes_per_file=0)

    Write images on disk with specified *writer* and file name *prefix*. Write
    to one file until the *bytes_per_file* bytes has been written. If it is 0,
    then one file per image is created. *writer* is a subclass of
    :class:`.writers.ImageWriter`. *start_index* specifies the number in the
    first file name, e.g. for the default *prefix* and *start_index* 100, the
    first file name will be image_00100.tif. If *prefix* is not formattable
    images are appended to the filename specified by *prefix*.
    """
    im_writer: Optional[TiffWriter] = None
    file_index = 0
    written = 0
    written_total = 0
    dir_name = os.path.dirname(prefix)
    # If there is no formatting user wants just one file, in which case we
    # append
    append = prefix.format(0) == prefix
    if append:
        im_writer = writer(prefix, bytes_per_file, append=True)

    if dir_name and not os.path.exists(dir_name):
        create_directory(dir_name, rights=rights)

    i = 0
    try:
        async for image in producer:
            if not append and (
                    not im_writer or written + image.nbytes > bytes_per_file):
                if im_writer:
                    im_writer.close()
                    LOG.debug('Writer "{}" closed'.format(prefix.format(start_index
                                                                        + file_index - 1)))
                im_writer = writer(prefix.format(start_index + file_index), bytes_per_file,
                                   first_frame=start_index+i)
                file_index += 1
                written = 0
            if im_writer:
                im_writer.write(image)
                written += image.nbytes
                written_total += image.nbytes
                i += 1
        return written_total
    finally:
        if im_writer:
            im_writer.close()
            LOG.debug('Writer "{}" closed'.format(
                prefix.format(start_index + file_index - 1)))


class Walker(AsyncObject):
    """
    A Walker moves through an abstract hierarchy and allows to write data
    at a specific location.
    """

    _root: str
    _current: str
    _log_name: str
    _lock: asyncio.Lock
    dsetname: str

    async def __ainit__(self,
                        root: str,
                        dsetname: str = "frames",
                        log_name: str = "experiment.log") -> None:
        """
        Constructor. *root* is the topmost level of the data structure

        :param root: topmost level of the data structure
        :type root: str
        :param dsetname: dataset name
        :type dsetname: str
        :param log_name: default log file name
        :type log_name: str
        """
        self._root = root
        self._log_name = log_name
        self._lock = asyncio.Lock()
        self.dsetname = dsetname
        await self.home()

    async def __aenter__(self) -> Walker:
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._lock.release()

    async def _descend(self, name: str) -> None:
        """Descend to *name*."""
        raise NotImplementedError

    async def _ascend(self) -> None:
        """Ascend from current depth."""
        raise NotImplementedError

    def _create_writer(self,
                       producer: AsyncIterable[ArrayLike],
                       dsetname: Optional[str] = None) -> Awaitable:
        """
        Subclass should provide the implementation for, how the writer should
        be created for asynchronously received data"""
        raise NotImplementedError

    async def _get_current(self) -> str:
        """Fetches the current from internal contex"""
        raise NotImplementedError

    async def home(self) -> None:
        """Return to root"""
        self._current = self._root

    @property
    async def current(self) -> str:
        """Return current position."""
        return await self._get_current()

    async def exists(self, *paths) -> bool:
        """Return True if path from current position specified by a list of
        *paths* exists."""
        raise NotImplementedError

    async def descend(self, name: str) -> Walker:
        """Descend to *name* and return *self*."""
        await self._descend(name)
        return self

    async def ascend(self) -> Walker:
        """Ascend from current depth and return *self*."""
        await self._ascend()
        return self

    async def create_writer(self,
                            producer: AsyncIterable[ArrayLike],
                            name: Optional[str] = None,
                            dsetname: Optional[str] = None) -> Awaitable:
        """
        Create a writer coroutine for writing data set *dsetname* with images
        from *producer* inside. If *name* is given, descend to it first and
        once the writer is created ascend back. This way, the writer can
        operate in *name* and the walker can be safely used to move around
        and create other writers elsewhere while the created writer is working.
        The returned coroutine is not guaranteed to be wrapped into a
        :class:`.asyncio.Task`, hence to be started immediately.  This function
        also does not block after creating the writer. This is useful for
        splitting the preparation of writing (creating directories, ...)
        and the I/O itself.
        """
        if name:
            await self.descend(name)
        try:
            return await self._create_writer(producer, dsetname=dsetname)
        finally:
            if name:
                await self.ascend()

    @background
    async def write(self,
                    producer: AsyncIterable[ArrayLike],
                    dsetname: Optional[str] = None) -> Awaitable:
        """
        Create a coroutine for writing data set *dsetname* with images from
        *producer*. The execution starts immediately in the background and
        await will block until the images are written.
        """
        return await self._create_writer(producer, dsetname=dsetname)

    async def get_log_handler(self) -> AsyncLoggingHandlerCloser:
        """Provides a log handler featuring an asynchronous flush and closure
        utility"""
        raise NotImplementedError

    @background
    async def log_to_json(self, payload: str) -> None:
        """
        Provides local counterpart of the remote logging of experiment
        metadata. Writes the provided payload to a static file called
        experiment.json.

        :param payload: content to write
        :type payload: str
        """
        raise NotImplementedError


class DummyWalker(Walker):
    """Walker object used for testing purposes"""

    _paths: Set[str]

    async def __ainit__(self, root: str = "") -> None:
        await super().__ainit__(root)
        self._paths = set([])

    @property
    async def paths(self) -> Iterable[str]:
        return self._paths

    async def exists(self, *paths) -> bool:
        return os.path.join(*paths) in self._paths

    async def _get_current(self) -> str:
        return self._current

    async def _descend(self, name) -> None:
        self._current = os.path.join(self._current, name)
        self._paths.add(self._current)

    async def _ascend(self) -> None:
        if self._current != self._root:
            self._current = os.path.dirname(self._current)

    def _create_writer(self,
                       producer: AsyncIterable[ArrayLike],
                       dsetname: Optional[str] = None) -> Awaitable:
        dsetname = dsetname or self.dsetname
        path = os.path.join(self._current, dsetname)

        async def _append_paths() -> None:
            i = 0
            async for item in producer:
                self._paths.add(os.path.join(path, str(i)))
                i += 1
        return _append_paths()

    async def get_log_handler(self) -> AsyncLoggingHandlerCloser:
        """Provides a no-op logging handler as a placeholder"""
        return NoOpLoggingHandler()


class DirectoryWalker(Walker):
    """
    A DirectoryWalker moves through a file system and writes flat files using a
    specific filename template.
    """

    _bytes_per_file: int
    _start_index: int
    writer: Type[TiffWriter]

    async def __ainit__(self,
                        root: Optional[str] = None,
                        dsetname: str = "frame_{:>06}.tif",
                        writer: Type[TiffWriter] = TiffWriter,
                        start_index: int = 0,
                        bytes_per_file: int = 0,
                        rights: str = "750") -> None:
        """
        Use *writer* to write data to files with filenames with a template
        from *dsetname*. *start_index* specifies the number in the first file
        name, e.g. for the default *dsetname* and *start_index* 100, the first
        file name will be frame_000100.tif.
        """
        # Handling root of the experimental file system
        if not root:
            root = os.getcwd()
        root = os.path.abspath(root)
        create_directory(root, rights=rights)
        self.writer = writer
        self._bytes_per_file = bytes_per_file
        self._start_index = start_index
        await super().__ainit__(root, dsetname)

    async def _descend(self, name: str) -> None:
        new = os.path.join(self._current, name)
        create_directory(new, rights=self._rights)
        self._current = new

    async def _ascend(self) -> None:
        if self._current == self._root:
            raise StorageError("Cannot break out of `{}'.".format(self._root))
        self._current = os.path.dirname(self._current)

    async def _get_current(self) -> str:
        """Provides current from local context"""
        return self._current

    def _create_writer(self,
                       producer: AsyncIterable[ArrayLike],
                       dsetname: Optional[str] = None) -> Awaitable:
        dsetname = dsetname or self.dsetname
        if self._dset_exists(dsetname):
            dset_prefix = split_dsetformat(dsetname)
            dset_path = os.path.join(self._current, dset_prefix)
            raise StorageError("`{}' is not empty".format(dset_path))
        prefix = os.path.join(self._current, dsetname)
        return write_images(
            producer,
            self.writer,
            prefix,
            self._start_index,
            self._bytes_per_file
        )

    def _dset_exists(self, dsetname: str) -> bool:
        """Check if *dsetname* exists on the current level."""
        if not re.match('.*{.*}.*', dsetname):
            raise ValueError('dsetname `{}\' has wrong format'.format(dsetname))
        filenames = os.listdir(self._current)
        for name in filenames:
            if name.startswith(split_dsetformat(dsetname)):
                return True
        return False

    async def exists(self, *paths: str) -> bool:
        """Check if *paths* exist."""
        return os.path.exists(os.path.join(await self.current, *paths))

    async def get_log_handler(self) -> AsyncLoggingHandlerCloser:
        return LoggingHandler(f"{await self.current}/{self._log_name}")

    async def log_to_json(self, payload: str) -> None:
        """
        Logs experiment metadata as *payload* to a file called experiment.json

        NOTE: This method does not have to be a coroutine. We still made it so
        to maintain coherence at the api level. With the unification of the
        top layer walker api this concern would be addressed.

        :param payload: content to write
        :type payload: str
        """
        with open(
                file=os.path.join(self._current, "experiment.json"),
                mode="w",
                encoding="utf-8") as lgf:
            lgf.write(payload)


class RemoteDirectoryWalker(Walker):
    """
    Defines the api layer of a directory walker for a remote file system.
    Encapsulates a Tango device which runs on the remote file system where the
    data needs to be written. Since it has th
    """

    device: RemoteDirectoryWalkerTangoDevice

    async def __ainit__(self,
                        device: RemoteDirectoryWalkerTangoDevice,
                        root: Optional[str] = None,
                        dsetname: str = "frame_{:>06}.tif",
                        wrt_cls: str = "TiffWriter",
                        start_index: int = 0,
                        bytes_per_file: int = 0) -> None:
        """
        Initializes a remote directory walker. This walker implementation
        encapsulates a Tango device server and delegates its core utilities
        to the same.
        :param device: an abstract tango device conforming to remote tango
        walker specification
        :type device: RemoteDirectoryWalkerTangoDevice
        :param root: file system root for to start traversal, if None current
        directory of the walker is used
        :type root: Optional[str]
        :param dsetname: template for writing files of the dataset
        :type dsetname: str
        :param wrt_cls: specific writer class which the device should use
        to write files
        :type wrt_cls: str
        :param start_index: number of the first file name in the dataset
        :type start_index: int
        :param bytes_per_file: size limit for a file, `0` denotes 1 file per
        image
        :type bytes_per_file: int
        """
        self.device = device
        LOG.debug("device attributes: %s", self.device.get_attribute_list())
        # The 'root' is either explicitly specified or initialized using the
        # value from the remote server where the Tango device is initialized
        # with reasonable defaults.
        if root:
            self._root = root
            await self.device.write_attribute(attr_name="root", value=root)
        else:
            self._root = (await self.device["root"]).value
        # Synchronizing internal state
        await self.home()
        await self.device.write_attribute(attr_name="writer_class",
                                          value=wrt_cls)
        await self.device.write_attribute(attr_name="dsetname", value=dsetname)
        await self.device.write_attribute(attr_name="start_index",
                                          value=start_index)
        await self.device.write_attribute(attr_name="bytes_per_file", value=bytes_per_file)
        await super().__ainit__(root=self._root, dsetname=dsetname)

    async def _descend(self, name: str) -> None:
        await self.device.descend(name)

    async def _ascend(self) -> None:
        if self._root == (await self.device["current"]).value:
            raise StorageError(f"cannot break out of {self._root}.")
        await self.device.ascend()

    async def _get_current(self):
        """Provides current from remote context"""
        return (await self.device["current"]).value

    async def home(self) -> None:
        """Return to root remotely and inside its own context (which is
        implemented in the super class)"""
        await self.device.write_attribute(attr_name="current", value=self._root)
        await super().home()

    async def exists(self, *paths: str) -> bool:
        """
        Asserts whether the specified paths exists in the file system.

        :param paths: a given number of file system paths
        :type paths: str
        :return: asserts whether specified path exists
        :rtype: bool
        """
        return await self.device.exists(*paths)

    async def create_writer(self,
                            producer: AsyncIterable[ArrayLike],
                            name: Optional[str] = None,
                            dsetname: Optional[str] = None) -> Awaitable:
        """
        Explicitly specifies that remote directory walker handles asynchronous
        data writing with a tango device server running remotely. Internally,
        corresponding device server uses the DirectoryWalker class to create
        the writer for incoming data.
        """
        raise NotImplementedError("delegates writing utility to remote tango server")

    @background
    async def write_sequence(self, path: str) -> None:
        """
        Asynchronously writes a sequence in the provided path

        :param path: path to write to
        :type path: str
        """
        await self.device.write_sequence(path)

    async def get_log_handler(self) -> AsyncLoggingHandlerCloser:
        """
        Provides a logging handler for the current path, capable to facilitate
        logging at a remote host.
        """
        await self.device.open_log_handler()
        return RemoteLoggingHandler(device=self.device)

    async def log_to_json(self, payload: str) -> None:
        """Implements api layer for writing experiment metadata"""
        await self.device.log_to_json(payload)


class StorageError(Exception):
    """Exception related to logical issues with storage"""
    ...


if __name__ == "__main__":
    pass
