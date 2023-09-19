"""Storage implementations."""
from __future__ import annotations
import asyncio
import os
import logging
import re
from typing import Optional, AsyncIterable, Awaitable, Type
from logging import FileHandler, Formatter
import tifffile
from concert.base import AsyncObject
from concert.coroutines.base import background
from concert.persistence.writers import TiffWriter
from concert.persistence.typing import RemoteDirectoryWalkerTangoDevice
from concert.persistence.typing import StorageError, ArrayLike
from concert.persistence.logger import RemoteLogger


LOG = logging.getLogger(__name__)

def split_dsetformat(dsetname):
    """
    Strip *dsetname* off the formatting part wihch leaves us with the data 
    set name.
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


def create_directory(directory, rights=0o0750):
    """Create *directory* and all paths along the way if necessary."""
    if not os.path.exists(directory):
        LOG.debug("Creating directory {}".format(directory))
        os.makedirs(directory, rights)


async def write_images(producer: AsyncIterable[ArrayLike], 
                       writer: Type[TiffWriter] = TiffWriter,
                       prefix: str = "image_{:>05}.tif",
                       start_index: int = 0,
                       bytes_per_file: int = 0) -> int:
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
    im_writer = None
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
        create_directory(dir_name)

    i = 0

    try:
        async for image in producer:
            if not append and (not im_writer or written + image.nbytes > bytes_per_file):
                if im_writer:
                    im_writer.close()
                    LOG.debug('Writer "{}" closed'.format(prefix.format(start_index
                                                                        + file_index - 1)))
                im_writer = writer(prefix.format(start_index + file_index), bytes_per_file)
                file_index += 1
                written = 0
            im_writer.write(image)
            written += image.nbytes
            written_total += image.nbytes
            i += 1

        return written_total
    finally:
        if im_writer:
            im_writer.close()
            LOG.debug('Writer "{}" closed'.format(prefix.format(start_index + file_index - 1)))


class Walker(object):

    """
    A Walker moves through an abstract hierarchy and allows to write data
    at a specific location.
    """

    def __init__(self, root, dsetname='frames', log=None, log_handler=None):
        """Constructor. *root* is the topmost level of the data structure."""
        self._root = root
        self._current = self._root
        self.dsetname = dsetname
        self._log = log
        self._log_handler = log_handler
        self._lock = asyncio.Lock()

        if self._log:
            self._log_handler.setLevel(logging.INFO)
            formatter = Formatter("[%(asctime)s] %(levelname)s: %(name)s: %(message)s")
            self._log_handler.setFormatter(formatter)
            self._log.addHandler(self._log_handler)

    def __del__(self):
        """Destructor."""
        if self._log:
            self._log_handler.close()
            self._log.removeHandler(self._log_handler)

    async def __aenter__(self):
        await self._lock.acquire()

        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._lock.release()

    def home(self):
        """Return to root."""
        self._current = self._root

    @property
    def current(self):
        """Return current position."""
        return self._current

    def exists(self, *paths):
        """Return True if path from current position specified by a list of *paths* exists."""
        raise NotImplementedError

    def descend(self, name):
        """Descend to *name* and return *self*."""
        self._descend(name)

        return self

    def ascend(self):
        """Ascend from current depth and return *self*."""
        self._ascend()

        return self

    def _descend(self, name):
        """Descend to *name*."""
        raise NotImplementedError

    def _ascend(self):
        """Ascend from current depth."""
        raise NotImplementedError

    def create_writer(self, producer, name=None, dsetname=None):
        """
        Create a writer coroutine for writing data set *dsetname* with images from *producer*
        inside. If *name* is given, descend to it first and once the writer is created ascend back.
        This way, the writer can operate in *name* and the walker can be safely used to move around
        and create other writers elsewhere while the created writer is working. The returned
        coroutine is not guaranteed to be wrapped into a :class:`.asyncio.Task`, hence to be started
        immediately.  This function also does not block after creating the writer. This is useful
        for splitting the preparation of writing (creating directories, ...) and the I/O itself.
        """
        if name:
            self.descend(name)

        try:
            return self._create_writer(producer, dsetname=dsetname)
        finally:
            if name:
                self.ascend()

    @background
    async def write(self, producer, dsetname=None):
        """
        Create a coroutine for writing data set *dsetname* with images from *producer*. The
        execution starts immediately in the background and await will block until the images are
        written.
        """
        return await self._create_writer(producer, dsetname=dsetname)


class RemoteWalker(AsyncObject):
    """
    A RemoteWalker moves through an abstract hierarchy and allows to
    asynchronously write data at a specific location.
    """

    _root: Optional[str]
    _current: Optional[str]
    dsetname: str
    _lock: asyncio.Lock

    async def __ainit__(self,
                        root: Optional[str], 
                        dsetname: str = "frames") -> None:
        """
        Initializes a remote walker encapsulating api-layer specifications
        for the same.

        :param root: optional file system root to start traversal
        :type root: Optional[str]
        :param dsetname: template or writing files of the dataset
        :type dsetname: str
        """
        self._root = root
        self._current = self._root
        self.dsetname = dsetname
        self._lock = asyncio.Lock()
        await super().__ainit__()

    async def __aenter__(self) -> RemoteWalker:
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._lock.release()

    def home(self) -> None:
        """Return to root."""
        self._current = self._root

    @property
    def current(self) -> str:
        """Return current position."""
        return self._current

    async def exists(self, *paths: str) -> bool:
        """
        Return True if path from current position specified by a list of 
        *paths* exists.
        """
        raise NotImplementedError

    async def descend(self, name: str) -> RemoteWalker:
        """Descend to *name* and return *self*."""
        await self._descend(name=name)
        return self

    async def ascend(self) -> RemoteWalker:
        """Ascend from current depth and return *self*."""
        await self._ascend()
        return self

    async def _descend(self, name: str) -> None:
        """Descend to *name*."""
        raise NotImplementedError

    async def _ascend(self) -> None:
        """Ascend from current depth."""
        raise NotImplementedError

    async def _create_writer(self,
                             producer: AsyncIterable[ArrayLike],
                             dsetname: Optional[str] = None) -> Awaitable:
        """Creates a writer coroutine to write images to disk"""
        raise NotImplementedError


    async def create_writer(self,
                            producer: AsyncIterable[ArrayLike],
                            name: Optional[str] = None,
                            dsetname: Optional[str] = None) -> Awaitable:
        """
        Create a writer coroutine for writing data set *dsetname* with images 
        from *producer* inside. If *name* is given, descend to it first and 
        once the writer is created ascend back. This way, the writer can 
        operate in *name* and the walker can be safely used to move around and 
        create other writers elsewhere while the created writer is working. 

        The returned coroutine is not guaranteed to be wrapped into a
        :class:`.asyncio.Task`, hence to be started immediately.  This function
        also does not block after creating the writer. This is useful for
        splitting the preparation of writing (creating directories, ...) and
        the I/O itself.
        """
        if name:
            await self.descend(name)
        try:
            return await self._create_writer(producer=producer,
                                             dsetname=dsetname)
        finally:
            if name:
                await self.ascend()

    @background
    async def write(self,
                    producer: AsyncIterable[ArrayLike],
                    dsetname: Optional[str] = None) -> int:
        """
        Creates a coroutine for writing data set *dsetname* with images from
        *producer*. The execution starts immediately in the background and
        await will block until the images are written. If successful, it
        returns the number of bytes written.
        """
        return await self._create_writer(producer, dsetname=dsetname)
    
    @background
    async def write_sequence(self, path: str) -> None:
        """
        Asynchronously writes a sequence in the provided path

        :param path: path to write to
        :type path: str
        """
        raise NotImplementedError


class DummyWalker(Walker):
    def __init__(self, root=''):
        super(DummyWalker, self).__init__(root)
        self._paths = set([])

    @property
    def paths(self):
        return self._paths

    def exists(self, *paths):
        return os.path.join(*paths) in self._paths

    def _descend(self, name):
        self._current = os.path.join(self._current, name)
        self._paths.add(self._current)

    def _ascend(self):
        if self._current != self._root:
            self._current = os.path.dirname(self._current)

    def _create_writer(self, producer, dsetname=None):
        dsetname = dsetname or self.dsetname
        path = os.path.join(self._current, dsetname)

        async def _append_paths():
            i = 0
            async for item in producer:
                self._paths.add(os.path.join(path, str(i)))
                i += 1

        return _append_paths()


class DirectoryWalker(Walker):

    """
    A DirectoryWalker moves through a file system and writes flat files using a
    specific filename template.
    """

    def __init__(self, writer=TiffWriter, dsetname='frame_{:>06}.tif', start_index=0,
                 bytes_per_file=0, root=None, log=None, log_name='experiment.log'):
        """
        Use *writer* to write data to files with filenames with a template from *dsetname*.
        *start_index* specifies the number in the first file name, e.g. for the default *dsetname*
        and *start_index* 100, the first file name will be frame_000100.tif.
        """
        if not root:
            root = os.getcwd()
        root = os.path.abspath(root)

        log_handler = None

        if log:
            create_directory(root)
            log_path = os.path.join(root, log_name)
            log_handler = FileHandler(log_path)

        super(DirectoryWalker, self).__init__(root, dsetname=dsetname,
                                              log=log, log_handler=log_handler)
        self.writer = writer
        self._bytes_per_file = bytes_per_file
        self._start_index = start_index

    def _descend(self, name):
        new = os.path.join(self._current, name)
        create_directory(new)
        self._current = new

    def _ascend(self):
        if self._current == self._root:
            raise StorageError("Cannot break out of `{}'.".format(self._root))

        self._current = os.path.dirname(self._current)

    def exists(self, *paths):
        """Check if *paths* exist."""
        return os.path.exists(os.path.join(self.current, *paths))

    def _create_writer(self, producer, dsetname=None):
        dsetname = dsetname or self.dsetname

        if self._dset_exists(dsetname):
            dset_prefix = split_dsetformat(dsetname)
            dset_path = os.path.join(self.current, dset_prefix)
            raise StorageError("`{}' is not empty".format(dset_path))

        prefix = os.path.join(self._current, dsetname)

        return write_images(
            producer,
            self.writer,
            prefix,
            self._start_index,
            self._bytes_per_file
        )

    def _dset_exists(self, dsetname):
        """Check if *dsetname* exists on the current level."""
        if not re.match('.*{.*}.*', dsetname):
            raise ValueError('dsetname `{}\' has wrong format'.format(dsetname))

        filenames = os.listdir(self._current)
        for name in filenames:
            if name.startswith(split_dsetformat(dsetname)):
                return True
        return False


class RemoteDirectoryWalker(RemoteWalker):
    """
    Defines the api layer of a directory walker for a remote file system.
    Encapsulates a Tango device which runs on the remote file system where the
    data needs to be written. Also encapsulates the frontend of remote logger
    to update the context for logging as the walker travseres the file system.
    """
    
    device: RemoteDirectoryWalkerTangoDevice
    _logger: Optional[RemoteLogger]

    async def __ainit__(self,
                        device: RemoteDirectoryWalkerTangoDevice,
                        wrt_cls: str = "TiffWriter",
                        dsetname: str = "frame_{:>06}.tif",
                        start_index: int = 0,
                        bytes_per_file: int = 0,
                        root: Optional[str] = None,
                        logger: Optional[RemoteLogger] = None) -> None:
        """
        Initializes a remote directory walker. This walker implementation
        encapsulates a Tango device server and delegates its core utilities
        to the same.

        :param device: an abstract tango device conforming to remote tango
        walker specification
        :type rem_uri: RemoteDirectoryWalkerTangoDevice
        :param wrt_cls: specific writer class which the device should use
        to write files
        :type wrt_cls: str
        :param dsetname: template for writing files of the dataset
        :type dsetname: str
        :param start_index: number of the first file name in the dataset
        :type start_index: int
        :param bytes_per_file: size limit for a file, `0` denotes 1 file per
        image
        :type bytes_per_file: int
        :param root: file system root for to start traversal, if None current
        directory of the walker is used
        :type root: Optional[str]
        :param logger: frontend object for remote logging utility
        :type log_dsp: Optional[RemoteLogger]
        """
        await super().__ainit__(root=root, dsetname=dsetname)
        self.device = device
        LOG.debug(
                f"Remote device attributes: {self.device.get_attribute_list()}")
        # If root is None, we initialize internal `root` and `current` values
        # of the api with respective values from the remote device.
        if root:
            self._root = root
            await self.device.write_attribute(attr_name="root",
                                              value=self._root)
        else:
            self._root = (await self.device["root"]).value
        self._current = self._root
        await self.device.write_attribute(attr_name="writer_class", 
                                          value=wrt_cls)
        await self.device.write_attribute(attr_name="dsetname", value=dsetname)
        await self.device.write_attribute(attr_name="start_index", 
                                          value=start_index)
        await self.device.write_attribute(
                attr_name="bytes_per_file", value=bytes_per_file)
        self._logger = logger
            
    async def _descend(self, name: str) -> None:
        await self.device.descend(name)
        self._current = (await self.device["current"]).value 
        if self._logger:
            self._logger.set_logging_path(new_path=self._current)

    async def _ascend(self) -> None:
        await self.device.ascend()
        self._current = (await self.device["current"]).value
        if self._logger:
            self._logger.set_logging_path(new_path=self._current)

    async def exists(self, *paths: str) -> bool:
        """
        Asserts whether the specified paths exists in the file system.

        :param paths: a given number of file system paths
        :type paths: str
        :return: asserts whether specified path exists
        :rtype: bool
        """
        return await self.device.exists(*paths)

    async def _create_writer(self,
                       producer: AsyncIterable[ArrayLike],
                       dsetname: Optional[str] = None) -> Awaitable:
        return await self.device.create_writer(producer=producer, 
                                               dsetname=dsetname)

    @background
    async def write_sequence(self, path: str) -> None:
        """
        Asynchronously writes a sequence in the provided path

        :param path: path to write to
        :type path: str
        """
        # NOTE: This approach fundamentally changes the RemoteWalker api since
        # we are trying to merge the writer and remote walker device servers.
        # Earlier writer device server behind the scene used to make a
        # DirectoryWalker and call its write method with the image data that
        # was asynchronously received. It worked because writer device server
        # behind the scene was a TangoRemoteProcessing object and in turn 
        # encapsulated the ZMQReceiver (in simpler terms data to be written is
        # available to the device server running at the remote host but not
        # with the current api layer object which possibly resides elsewhere).
        # Hence, with this decentralized approach all the methods of the
        # RemoteWalker class that accepts a `producer: AsyncIterable[ArrayLike]`
        # are potentially pointless. We therefore consider to make the
        # RemoteWalker api different from Walker(local) api.
        # As a side note, when our remote logger api is ready we consider to
        # remove logging-related components from RemoteWalker because it is
        # unnecessary. We want to separate that concern from the one of the
        # remote walker.
        await self.device.write_sequence(path)


if __name__ == "__main__":
    pass

