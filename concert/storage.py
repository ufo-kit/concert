"""Storage implementations."""
from concert.ext import tifffile
from concert.asynchronous import async


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
