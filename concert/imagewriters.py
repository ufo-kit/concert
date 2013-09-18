"""Image writer implementations."""
from concert.ext import tifffile


def write_tiff(file_name_prefix, data):
    """The default TIFF writer which uses tifffile module."""
    tifffile.imsave(file_name_prefix.join(".tif"), data)


def write_libtiff(file_name_prefix, data):
    """Write a TIFF file using pylibtiff."""
    from libtiff import TIFF

    tiff_file = TIFF.open(file_name_prefix.join(".tif"), "w")
    try:
        tiff_file.write_image(data)
    finally:
        tiff_file.close()
