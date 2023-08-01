"""Storage implementations."""
from __future__ import annotations

import asyncio
import os
import logging
from typing import Awaitable, Optional, Type, Dict, Callable, AsyncIterable
from logging import Logger, Handler
import re
from logging import FileHandler, Formatter
import tifffile  # type: ignore[import]
import numpy as np
from concert.coroutines.base import background
from concert.writers import TiffWriter

LOG = logging.getLogger(__name__)


def read_tiff(file_name: str) -> np.ndarray:
    """Defines a recommended reader for TIFF image files using :py:mod:`tifffile` module.

    :param file_name: tiff file name
    :type file_name: str
    :returns: tiff image file as ndarray
    :rtype: np.ndarray
    """
    with tifffile.TiffFile(file_name) as f:
        return f.asarray(out="memmap")


READERS: Dict[str, Callable[[str], np.ndarray]] = {".tif": read_tiff, ".tiff": read_tiff}

# Imports optional package fabio to read EDF(extended depth of filed) format images
# if applicable
try:
    import fabio  # type: ignore[import]

    def read_edf_via_fabio(filename: str) -> np.ndarray:
        """Defines an optional reader function to read the images in EDF(extended depth of field)
        format and return as ndarray

        :param filename: edf image file name
        :type filename: str
        :returns: edf image file as ndarray
        :rtype: np.ndarray
        """
        edf = fabio.edfimage.edfimage()
        edf.read(filename)
        return edf.data

    for ext in (".edf", ".edf.gz"):
        READERS[ext] = read_edf_via_fabio
except ImportError:
    LOG.debug("could not import fabio package as optional dependency")
    pass


def read_image(filename: str) -> np.ndarray:
    """
    Read image from file with *filename*. The file type is detected
    automatically.

    :param filename: image file name
    :type filename: str
    :returns: image file as ndarray
    :rtype: np.ndarray
    """
    for extension, reader in list(READERS.items()):
        if filename.lower().endswith(extension):
            return reader(filename)
    raise ValueError("Unsupported file type")


def write_tiff(file_name: str, data) -> str:
    """
    The default TIFF writer which uses :py:mod:`tifffile` module.
    Return the written file name.
    """
    tifffile.imwrite(file_name, data)
    return file_name


def write_libtiff(file_name, data):
    """Write a TIFF file using pylibtiff. Return the written file name."""
    from libtiff import TIFF  # type: ignore[import]

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


async def write_images(
        producer: AsyncIterable[np.ndarray],
        writer: Type[TiffWriter] = TiffWriter,
        prefix: str = "image_{:>05}.tif",
        start_index: int = 0,
        bytes_per_file: int = 0) -> int:
    """
    write_images(pqueue, writer=TiffWriter, prefix="image_{:>05}.tif", start_index=0,
                 bytes_per_file=0)

    Write images on disk with specified *writer* and file name *prefix*. Write to one file until the
    *bytes_per_file* bytes has been written. If it is 0, then one file per image is created.
    *writer* is a subclass of :class:`.writers.ImageWriter`. *start_index* specifies the number in
    the first file name, e.g. for the default *prefix* and *start_index* 100, the first file name
    will be image_00100.tif. If *prefix* is not format-able images are appended to the filename
    specified by *prefix*.

    :param producer: asynchronous iterable object for slice volumes
    :type producer: AsyncIterable[np.ndarray]
    :param writer: tiff image writer utility
    :type writer: TiffWriter
    :param prefix: image file name prefix
    :type prefix: str
    :param start_index: index number of the first file name, defaults to 0
    :type start_index: int
    :param bytes_per_file: bytes to write in one file, defaults to 0, means 1 file per image is
    created
    :type bytes_per_file: int
    :returns: total bytes written
    :rtype: int
    """
    im_writer = None
    file_index = 0
    written = 0
    written_total = 0
    dir_name = os.path.dirname(prefix)
    # If there is no formatting user wants just one file, in which case we append
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
            if im_writer:
                im_writer.write(image)
            written += image.nbytes
            written_total += image.nbytes
            i += 1
        return written_total
    finally:
        if im_writer:
            im_writer.close()
            LOG.debug('Writer "{}" closed'.format(prefix.format(start_index + file_index - 1)))


def split_dsetformat(dsetname: str) -> str:
    """Strip *dsetname* off the formatting part which leaves us with the data set name."""
    return dsetname.split("{")[0]


class StorageError(Exception):
    """Exceptions related to logical issues with storage."""
    pass


class Walker(object):
    """
    A Walker moves through an abstract hierarchy and allows to write data
    at a specific location.

    - **parameters**, **types**, **return** and **return types**::

        :param _root: top level of the data structure
        :type _root: str
        :param _current: current position
        :type _current: str
        :param dsetname: dataset name
        :type dsetname: str
        :param _log: logger for walker
        :type _log: logging.Logger
        :param _log_handler: logging event dispatcher
        :type _log_handler: logging.Handler
    """

    def __init__(self,
                 root: str,
                 dsetname: str = 'frames',
                 log: Optional[Logger] = None,
                 log_handler: Optional[Handler] = None) -> None:
        """Constructor. *root* is the topmost level of the data structure."""
        self._root = root
        self._current = self._root
        self.dsetname = dsetname
        self._log = log
        self._log_handler = log_handler
        self._lock = asyncio.Lock()
        if self._log and self._log_handler:
            self._log_handler.setLevel(logging.INFO)
            formatter = Formatter("[%(asctime)s] %(levelname)s: %(name)s: %(message)s")
            self._log_handler.setFormatter(formatter)
            self._log.addHandler(self._log_handler)

    def __del__(self) -> None:
        """Destructor."""
        if self._log and self._log_handler:
            self._log_handler.close()
            self._log.removeHandler(self._log_handler)

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._lock.release()

    def home(self) -> None:
        """Return to root."""
        self._current = self._root

    @property
    def current(self) -> str:
        """Return current position."""
        return self._current

    def exists(self, *paths: str) -> bool:
        """Return True if path from current position specified by a list of *paths* exists."""
        raise NotImplementedError

    def descend(self, name: str) -> Walker:
        """Descend to *name* and return *self*."""
        self._descend(name)
        return self

    def ascend(self) -> Walker:
        """Ascend from current depth and return *self*."""
        self._ascend()
        return self

    def _descend(self, name) -> None:
        """Descend to *name*."""
        raise NotImplementedError

    def _ascend(self) -> None:
        """Ascend from current depth."""
        raise NotImplementedError

    def _create_writer(self,
                       producer: AsyncIterable[np.ndarray],
                       dsetname: Optional[str] = None) -> Awaitable:
        """Creates a coroutines for writing data set

        :param producer: asynchronous iterable object of slice volumes
        :type producer: AsyncIterable[np.ndarray]
        :param dsetname: dataset name
        :type dsetname: Optional[str]
        :returns: Coroutine object to create a writer
        :rtype: Awaitable
        """
        raise NotImplementedError

    def create_writer(self,
                      producer: AsyncIterable[np.ndarray],
                      name: Optional[str] = None,
                      dsetname: Optional[str] = None) -> Awaitable:
        """
        Create a writer coroutine for writing data set *dsetname* with images from *producer*
        inside. If *name* is given, descend to it first and once the writer is created ascend back.
        This way, the writer can operate in *name* and the walker can be safely used to move around
        and create other writers elsewhere while the created writer is working. The returned
        coroutine is not guaranteed to be wrapped into a :class:`.asyncio.Task`, hence to be started
        immediately.  This function also does not block after creating the writer. This is useful
        for splitting the preparation of writing (creating directories, ...) and the I/O itself.

        :param producer: asynchronous iterable object of slice volumes
        :type producer: AsyncIterable[np.ndarray]
        :param name: a directory path for writing data
        :type name: Optional[str]
        :param dsetname: image dataset name
        :type dsetname: Optional[str]
        :returns: Awaitable writer coroutine
        :rtype: Awaitable
        """
        if name:
            self.descend(name)
        try:
            return self._create_writer(producer, dsetname=dsetname)
        finally:
            if name:
                self.ascend()

    @background
    async def write(self,
                    producer: AsyncIterable[np.ndarray],
                    dsetname: Optional[str] = None) -> Awaitable:
        """
        Create a coroutine for writing data set *dsetname* with images from *producer*. The
        execution starts immediately in the background and await will block until the images are
        written.
        """
        return await self._create_writer(producer, dsetname=dsetname)


class DirectoryWalker(Walker):
    """
    A DirectoryWalker moves through a file system and writes flat files using a
    specific filename template.
    """

    def __init__(self,
                 writer: Type[TiffWriter] = TiffWriter,
                 dsetname: str = "frame_{:>06}.tif",
                 start_index: int = 0,
                 bytes_per_file: int = 0,
                 root: Optional[str] = None,
                 log: Optional[Logger] = None,
                 log_name: str = "experiment.log") -> None:
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

    def _descend(self, name: str) -> None:
        new = os.path.join(self._current, name)
        create_directory(new)
        self._current = new

    def _ascend(self) -> None:
        if self._current == self._root:
            raise StorageError("Cannot break out of `{}'.".format(self._root))
        self._current = os.path.dirname(self._current)

    def exists(self, *paths: str) -> bool:
        """Check if *paths* exist."""
        return os.path.exists(os.path.join(self.current, *paths))

    def _create_writer(self,
                       producer: AsyncIterable[np.ndarray],
                       dsetname: Optional[str] = None) -> Awaitable:
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

    def _dset_exists(self, dsetname: str) -> bool:
        """Check if *dsetname* exists on the current level."""
        if not re.match('.*{.*}.*', dsetname):
            raise ValueError('dsetname `{}\' has wrong format'.format(dsetname))
        filenames = os.listdir(self._current)
        for name in filenames:
            if name.startswith(split_dsetformat(dsetname)):
                return True
        return False


if __name__ == "__main__":
    pass
