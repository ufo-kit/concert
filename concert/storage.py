"""Storage implementations."""
from concert.ext import tifffile
from concert.asynchronous import async
import os


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
    """The default TIFF writer which uses :py:mod:`tifffile` module."""
    tifffile.imsave(file_name_prefix + ".tif", data)


@async
def write_libtiff(file_name_prefix, data):
    """Write a TIFF file using pylibtiff."""
    from libtiff import TIFF

    tiff_file = TIFF.open(file_name_prefix + ".tif", "w")
    try:
        tiff_file.write_image(data)
    finally:
        tiff_file.close()
