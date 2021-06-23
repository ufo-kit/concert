"""Image writers for uniform acces by :func:`.storage.write_images`"""


class ImageWriter(object):
    def __init__(self, filename, bytes_per_file, append=False):
        self._writer = None

    def write(self, image):
        raise NotImplementedError

    def close(self):
        self._writer.close()


class TiffWriter(ImageWriter):
    def __init__(self, filename, bytes_per_file, append=False):
        import tifffile
        # 2 ** 25 from tifffile
        self._writer = tifffile.TiffWriter(filename, append=append,
                                           bigtiff=bytes_per_file >= 2 ** 32 - 2 ** 25)

    def write(self, image):
        self._writer.save(image)


class LibTiffWriter(ImageWriter):
    def __init__(self, filename, bytes_per_file, append=False):
        from libtiff import TIFF
        mode = 'a' if append else 'w'
        if bytes_per_file >= 2 ** 31:
            mode += '8'
        self._writer = TIFF.open(filename, mode)

    def write(self, image):
        self._writer.write_image(image)
