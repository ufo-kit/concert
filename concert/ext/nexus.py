"""
NeXus file format related utitilities.

This module provides convenience utilities to create NeXus-compatible data
sets. It uses NeXpy_ to interface with NeXus.

.. _NeXpy: http://wiki.nexusformat.org/NeXpy
"""
from logging import StreamHandler
from concert.storage import Walker, StorageError
from concert.coroutines.base import coroutine

try:
    import h5py
except ImportError:
    print("h5py is not installed")


class Hdf5Walker(Walker):

    """An HDF5 file walker implementation."""

    def __init__(self, hdf5, dsetname='frames', log=None, log_name='log'):
        """
        *hdf5* is a writeable h5py.File file. *fname* is the dataset name that the
        sequence is stored in.
        """
        log_handler = None
        if log:
            dtype = h5py.special_dtype(vlen=bytes)
            # Append to log if existing
            if log_name not in hdf5:
                log_dset = hdf5.create_dataset(log_name, (1,), maxshape=(None,), dtype=dtype)
            else:
                log_dset = hdf5[log_name]
            log_handler = StreamHandler(stream=Hdf5Stream(log_dset))

        super(Hdf5Walker, self).__init__(hdf5, dsetname=dsetname, log=log, log_handler=log_handler)

    def _descend(self, name):
        if self.exists(name):
            self._current = self._current[name]
        else:
            self._current = self._current.create_group(name)

    def _ascend(self):
        self._current = self._current.parent

    def exists(self, *paths):
        """Check if *paths* exist."""
        return '/'.join(paths) in self.current

    @coroutine
    def _write_coroutine(self, dsetname=None):
        """Write frames to data set *dsetname*."""
        data = yield
        shape = (1, ) + data.shape
        maxshape = (None, ) + data.shape
        dsetname = dsetname or self.dsetname

        if dsetname in self._current:
            raise StorageError("`{}' is not empty".format(self._current.name + '/' + dsetname))

        dset = self._current.create_dataset(dsetname, shape, maxshape=maxshape, dtype=data.dtype)

        dset[0, :, :] = data[:, :]
        i = 1

        while True:
            data = yield
            shape = (i + 1, ) + shape[1:]
            dset.resize(shape)
            dset[i, :, :] = data[:, :]
            i += 1


class Hdf5Stream(object):
    """A stream for logging output into an hdf5 file. *dataset* is the dataset of the file to which
    the log output will be written as plain text. The dataset must be resizable and accept plain
    text.
    """

    def __init__(self, dataset):
        """Constructor."""
        self._dset = dataset
        self._iteration = dataset.shape[0] - 1

    def write(self, data):
        """Write logging *data*."""
        self._dset[self._iteration] = data
        self._iteration += 1
        self._dset.resize((self._iteration + 1,))

    def flush(self):
        """flush the stream to the file."""
        self._dset.file.flush()
