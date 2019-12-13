"""Image writers for uniform acces by :func:`.storage.write_images`"""


class ImageWriter(object):
    def __init__(self, filename, bytes_per_file):
        self._writer = None

    def write(self, image):
        raise NotImplementedError

    def close(self):
        self._writer.close()


class TiffWriter(ImageWriter):
    def __init__(self, filename, bytes_per_file):
        import tifffile
        self._writer = tifffile.TiffWriter(filename, bigtiff=bytes_per_file >= 2 ** 31)

    def write(self, image):
        self._writer.save(image)


class LibTiffWriter(ImageWriter):
    def __init__(self, filename, bytes_per_file):
        from libtiff import TIFF
        self._writer = TIFF.open(filename, 'w8' if bytes_per_file >= 2 ** 31 else 'w')

    def write(self, image):
        self._writer.write_image(image)
