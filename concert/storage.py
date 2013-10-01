"""Storage implementations."""
import os
import logbook
from concert.ext import tifffile
from concert.helpers import async


LOGGER = logbook.Logger(__name__)


@async
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


@async
def write_tiff(file_name_prefix, data):
    """
    The default TIFF writer which uses :py:mod:`tifffile` module.
    Return the written file name.
    """
    file_name = file_name_prefix + ".tif"
    tifffile.imsave(file_name, data)

    return file_name


@async
def write_libtiff(file_name_prefix, data):
    """Write a TIFF file using pylibtiff. Return the written file name."""
    from libtiff import TIFF

    file_name = file_name_prefix + ".tif"
    tiff_file = TIFF.open(file_name, "w")
    try:
        tiff_file.write_image(data)
    finally:
        tiff_file.close()

    return file_name


def create_folder(folder, rights=0750):
    """Create *folder* and all paths along the way if necessary."""
    if not os.path.exists(folder):
        LOGGER.info("Creating folder {}".format(folder))
        os.makedirs(folder, rights)
