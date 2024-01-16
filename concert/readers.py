"""Image readers for convenient work with multi-page image sequences."""
import ast
import glob
import json
import os
from concert.coroutines.base import run_in_executor
from concert.helpers import ImageWithMetadata


class FileSequenceReader:

    """Image sequence reader optimized for reading consecutive images. One multi-page image file is
    not closed after an image is read so that it does not have to be re-opened for reading the next
    image. The :func:`.close` function must be called explicitly in order to close the last opened
    image.
    """

    def __init__(self, file_prefix, ext=''):
        if os.path.isdir(file_prefix):
            file_prefix = os.path.join(file_prefix, '*' + ext)
        self._filenames = sorted(glob.glob(file_prefix))
        if not self._filenames:
            raise SequenceReaderError("No files matching `{}' found".format(file_prefix))
        self._lengths = {}
        self._file = None
        self._filename = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    async def read_range(self, start=0, stop=None, step=1):
        if stop is None:
            stop = self.num_images
        if stop > self.num_images:
            raise SequenceReaderError('Stop greater than number of images')

        try:
            for i in range(start, stop, step):
                yield await run_in_executor(self.read, i)
        finally:
            self.close()

    @property
    def num_images(self):
        num = 0
        for filename in self._filenames:
            num += self._get_num_images_in_file(filename)

        return num

    def read(self, index):
        if index < 0:
            # Enables negative indexing
            index += self.num_images
        file_index = 0
        while index >= 0:
            if file_index >= len(self._filenames):
                raise SequenceReaderError('image index greater than sequence length')
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

    def _close_real(self):
        """Closes the open file."""
        raise NotImplementedError

    def _get_num_images_in_file_real(self):
        raise NotImplementedError

    def _read_real(self, index):
        raise NotImplementedError


class TiffSequenceReader(FileSequenceReader):
    def __init__(self, file_prefix, ext='.tif'):
        self._metadata_file = None
        super(TiffSequenceReader, self).__init__(file_prefix, ext=ext)

    def _open_real(self, filename):
        metadata_file_name = os.path.splitext(filename)[0] + ".json"
        if os.path.exists(metadata_file_name):
            with open(metadata_file_name, 'r') as f:
                self._json_metadata = json.load(f)
        import tifffile
        return tifffile.TiffFile(filename)

    def _close_real(self):
        self._file.close()

    def _get_num_images_in_file_real(self):
        return len(self._file.pages)

    def _read_real(self, index):
        image = self._file.pages[index].asarray().view(ImageWithMetadata)
        if self._metadata_file:
            image.metadata = self._json_metadata[str(index)]
        else:
            try:
                image.metadata = ast.literal_eval(self._file.pages[index].description)
            except SyntaxError:
                # No metadata in file
                pass
            except Exception as e:
                raise e
        return image


class SequenceReaderError(Exception):
    pass
