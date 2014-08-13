"""Storage implementations."""
import os
import logging
from logging import FileHandler, Formatter
from concert.ext import tifffile
from concert.coroutines.base import coroutine, inject


LOG = logging.getLogger(__name__)


def read_image(file_name):
    """
    Read image from file with *file_name*. The file type is detected
    automatically.
    """
    ext = os.path.splitext(file_name)[1]
    if ext in READERS:
        return READERS[ext](file_name)

    raise ValueError("Unsupported file type")


def read_tiff(file_name):
    """Read tiff file from disk by :py:mod:`tifffile` module."""
    return tifffile.imread(file_name)


READERS = {".tif": read_tiff,
           ".tiff": read_tiff}


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


@coroutine
def write_images(writer=write_tiff, prefix="image_{:>05}.tif"):
    """
    write_images(writer=write_tiff, prefix="image_{:>05}.tif")

    Write images on disk with specified *writer* and file name *prefix*.
    *writer* is a callable with the following nomenclature::

        writer(file_name, data)
    """
    i = 0
    dir_name = os.path.dirname(prefix)

    if dir_name and not os.path.exists(dir_name):
        create_directory(dir_name)

    while True:
        data = yield
        writer(prefix.format(i), data)
        i += 1


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

    def write(self, data=None, dsetname=None):
        """Write a sequence of *data* if specified, otherwise this method turns into a coroutine.
        The data set name is given by *dsetname*.
        """
        write_coro = self._write_coroutine(dsetname=dsetname)

        if data is None:
            return write_coro
        else:
            inject(data, write_coro)

    def _write_coroutine(self, dsetname=None):
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

    @coroutine
    def _write_coroutine(self, dsetname=None):
        dsetname = dsetname or self.dsetname
        path = os.path.join(self._current, dsetname)

        i = 0
        while True:
            yield
            self._paths.add(os.path.join(path, str(i)))
            i += 1


class DirectoryWalker(Walker):

    """
    A DirectoryWalker moves through a file system and writes flat files using a
    specific filename template.
    """

    def __init__(self, write_func=write_tiff, dsetname='frame_{:>06}.tif', root=None,
                 log=None, log_name='experiment.log'):
        """
        Use *write_func* to write data to files with filenames with a template
        from *dsetname*.
        """
        if not root:
            root = os.getcwd()

        log_handler = None

        if log:
            create_directory(root)
            log_path = os.path.join(root, log_name)
            log_handler = FileHandler(log_path)

        super(DirectoryWalker, self).__init__(root, dsetname=dsetname,
                                              log=log, log_handler=log_handler)
        self._write_func = write_func

    def _descend(self, name):
        self._current = os.path.join(self._current, name)
        create_directory(self._current)

    def _ascend(self):
        if self._current == self._root:
            raise StorageError("Cannot break out of `{}'.".format(self._root))

        self._current = os.path.dirname(self._current)

    def exists(self, *paths):
        """Check if *paths* exist."""
        return os.path.exists(os.path.join(self.current, *paths))

    def _write_coroutine(self, dsetname=None):
        dsetname = dsetname or self.dsetname

        if self._dset_exists(dsetname):
            dset_prefix = split_dsetformat(dsetname)
            dset_path = os.path.join(self.current, dset_prefix)
            raise StorageError("`{}' is not empty".format(dset_path))

        prefix = os.path.join(self._current, dsetname)
        return write_images(writer=self._write_func, prefix=prefix)

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
