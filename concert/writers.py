"""Image writers for uniform acces by :func:`.storage.write_images`"""
import json

from concert.helpers import ImageWithMetadata
import os
from concert import config


class ImageWriter:
    def __init__(self, filename, bytes_per_file, first_frame=0, append=False, metadata_file=True):
        if config.ALWAYS_WRITE_JSON_METADATA_FILE:
            metadata_file = True
        self._writer = None
        self._frame_number = first_frame
        self._metadata_file = None
        self._append = append
        if metadata_file:
            self._metadata_filename = self._metadata_file_name = os.path.splitext(filename)[
                                                                     0] + ".json"
            self._metadata_file = open(self._metadata_file_name, 'a')
            self._metadata_file.write("{\n")
            self._first_entry = True

    def write(self, image):
        self._write_real(image)
        self._write_metadata(image)
        self._frame_number += 1

    def _write_real(self, image):
        raise NotImplementedError

    def _write_metadata(self, image):
        if self._metadata_file:
            metadata = {} if not isinstance(image, ImageWithMetadata) else image.metadata
            metadata_json = json.dumps(metadata)
            if not self._first_entry:
                self._metadata_file.write(",\n")
            else:
                self._first_entry = False
            self._metadata_file.write(f'"{self._frame_number}": {metadata_json}')

    def close(self):
        self._writer.close()
        if self._metadata_file:
            self._metadata_file.write("}\n")
            self._metadata_file.close()


class TiffWriter(ImageWriter):
    def __init__(self, filename, bytes_per_file, first_frame=0, append=False):
        write_metadata_file = append
        # If we use append = True, the metadata will be stored in a separate file instead of the
        # tiff metadata.
        super().__init__(filename, bytes_per_file, first_frame=first_frame, append=append,
                         metadata_file=write_metadata_file)
        import tifffile
        # 2 ** 25 from tifffile
        self._writer = tifffile.TiffWriter(filename, append=append,
                                           bigtiff=bytes_per_file >= 2 ** 32 - 2 ** 25)

    def _write_real(self, image):
        if self._append:
            metadata = {}
        else:
            metadata = {} if not isinstance(image, ImageWithMetadata) else image.metadata
        self._writer.write(image, metadata=metadata)
