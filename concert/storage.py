"""Storage implementations."""
import contextlib
import os
import logging
import tifffile
from logging import FileHandler, Formatter
from concert.coroutines.base import background, feed_queue
from concert.writers import TiffWriter


LOG = logging.getLogger(__name__)


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


def write_images(pqueue, writer=TiffWriter, prefix="image_{:>05}.tif", start_index=0,
                 bytes_per_file=0):
    """
    write_images(pqueue, writer=TiffWriter, prefix="image_{:>05}.tif", start_index=0,
                 bytes_per_file=0)

    Write images on disk with specified *writer* and file name *prefix*. Write to one file until the
    *bytes_per_file* bytes has been written. If it is 0, then one file per image is created.
    *writer* is a subclass of :class:`.writers.ImageWriter`. *start_index* specifies the number in
    the first file name, e.g. for the default *prefix* and *start_index* 100, the first file name
    will be image_00100.tif. If *prefix* is not formattable images are appended to the filename
    specified by *prefix*.
    """
    im_writer = None
    file_index = 0
    written = 0
    dir_name = os.path.dirname(prefix)
    # If there is no formatting user wants just one file, in which case we append
    append = prefix.format(0) == prefix
    if append:
        im_writer = writer(prefix, bytes_per_file, append=True)

    if dir_name and not os.path.exists(dir_name):
        create_directory(dir_name)

    i = 0

    try:
        while True:
            image = pqueue.get().data
            if image is None:
                pqueue.task_done()
                break
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
            i += 1
            pqueue.task_done()
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

    @contextlib.contextmanager
    def inside(self, name):
        self.descend(name)
        try:
            yield
        finally:
            self.ascend()

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

    @background
    async def write(self, producer, dsetname=None):
        """Coroutine for writing data set *dsetname*."""
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

    @background
    async def write(self, producer, dsetname=None):
        dsetname = dsetname or self.dsetname
        path = os.path.join(self._current, dsetname)

        i = 0
        async for item in producer:
            self._paths.add(os.path.join(path, str(i)))
            i += 1


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
        self._writer = writer
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

    @background
    async def write(self, producer, dsetname=None):
        dsetname = dsetname or self.dsetname

        if self._dset_exists(dsetname):
            dset_prefix = split_dsetformat(dsetname)
            dset_path = os.path.join(self.current, dset_prefix)
            raise StorageError("`{}' is not empty".format(dset_path))

        prefix = os.path.join(self._current, dsetname)

        return await feed_queue(producer, write_images, self._writer, prefix,
                                self._start_index, self._bytes_per_file)

    def _dset_exists(self, dsetname):
        """Check if *dsetname* exists on the current level."""
        bad = '{' not in dsetname

        try:
            dsetname.format(0)
        except ValueError:
            bad = True

        if bad:
            raise ValueError('dsetname `{}\' has wrong format'.format(dsetname))

        filenames = os.listdir(self._current)
        for name in filenames:
            if name.startswith(split_dsetformat(dsetname)):
                return True

        return False


def split_dsetformat(dsetname):
    """Strip *dsetname* off the formatting part wihch leaves us with the data set name."""
    return dsetname.split('{')[0]


class StorageError(Exception):
    """Exceptions related to logical issues with storage."""
    pass
