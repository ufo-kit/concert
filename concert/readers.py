"""Image readers for convenient work with multi-page image sequences."""
import glob
import os


class FileSequenceReader(object):

    """Image sequence reader optimized for reading consecutive images. One multi-page image file is
    not closed after an image is read so that it does not have to be re-opened for reading the next
    image. The :func:`.close` function must be called explicitly in order to close the last opened
    image.
    """

    def __init__(self, file_prefix, ext=''):
        if os.path.isdir(file_prefix):
            file_prefix = os.path.join(file_prefix, '*' + ext)
        self._filenames = sorted(glob.glob(file_prefix))
        self._lengths = {}
        self._file = None
        self._filename = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def num_images(self):
        num = 0
        for filename in self._filenames:
            num += self._get_num_images_in_file(filename)

        return num

    def read(self, index):
        file_index = 0
        while index >= 0:
            index -= self._get_num_images_in_file(self._filenames[file_index])
            file_index += 1

        file_index -= 1
        index += self._lengths[self._filenames[file_index]]
        self._open(self._filenames[file_index])

        return self._read_real(index)

    def _open(self, filename):
        if self._filename != filename:
            if self._filename:
                self.close()
            self._file = self._open_real(filename)
            self._filename = filename

    def close(self):
        if self._filename:
            self._close_real()
            self._file = None
            self._filename = None

    def _get_num_images_in_file(self, filename):
        if filename not in self._lengths:
            self._open(filename)
            self._lengths[filename] = self._get_num_images_in_file_real()

        return self._lengths[filename]

    def _open_real(self, filename):
        """Returns an open file."""
        raise NotImplementedError

    def _close_real(self, filename):
        raise NotImplementedError

    def _get_num_images_in_file_real(self):
        raise NotImplementedError

    def _read_real(self, index):
        raise NotImplementedError


class TiffSequenceReader(FileSequenceReader):
    def __init__(self, file_prefix, ext='.tif'):
        super(TiffSequenceReader, self).__init__(file_prefix, ext=ext)

    def _open_real(self, filename):
        import tifffile
        return tifffile.TiffFile(filename)

    def _close_real(self):
        self._file.close()

    def _get_num_images_in_file_real(self):
        return len(self._file.pages)

    def _read_real(self, index):
        return self._file.pages[index].asarray()


class SequenceReaderError(Exception):
    pass
