import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
from concert import config
from concert.tests import TestCase
from concert.coroutines.base import async_generate
from concert.helpers import ImageWithMetadata
from concert.writers import TiffWriter
from concert.readers import TiffSequenceReader
from concert.storage import DirectoryWalker


def generate_frames(number=10, dimensions=(10, 10)) -> [ImageWithMetadata]:
    frames = []
    for i in range(number):
        image = np.random.random(dimensions)
        image = image.astype(np.float32)
        metadata = {"time": datetime.now().isoformat(), "random_value": np.random.random()}
        image = ImageWithMetadata(image, metadata=metadata)
        frames.append(image)
    return frames


class TestReaderWriter(TestCase):
    async def asyncSetUp(self) -> None:
        self._data_dir = tempfile.mkdtemp()
        print(self._data_dir)
        self.data = generate_frames(20)

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir)

    async def run_test(self, enforce_json_file=False, bytes_per_file=0):
        config.ALWAYS_WRITE_JSON_METADATA_FILE = enforce_json_file
        run_file_name = datetime.now().strftime("%y%m%d_%H%M%S.%f")
        folder = os.path.join(self._data_dir, run_file_name)
        os.mkdir(folder)
        walker = DirectoryWalker(root=folder,
                                 writer=TiffWriter,
                                 dsetname="frame_{:>06}.tif",
                                 bytes_per_file=bytes_per_file)
        await walker.write(async_generate(self.data))
        reader = TiffSequenceReader(os.path.join(folder))
        data2 = []
        num_frames = reader.num_images
        for i in range(num_frames):
            data2.append(reader.read(index=i))
        self.compare_frames(self.data, data2)

    def compare_frames(self, frames_01: [ImageWithMetadata],
                       frames_02: [ImageWithMetadata]) -> None:
        self.assertEqual(len(frames_01), len(frames_02))
        for i in range(len(frames_01)):
            self.assertTrue(np.array_equal(frames_01[i], frames_02[i]))
            self.assertEqual(frames_01[i].metadata["time"], frames_02[i].metadata["time"])
            self.assertEqual(frames_01[i].metadata["random_value"],
                             frames_02[i].metadata["random_value"])

    async def test_tifffile(self) -> None:
        await self.run_test(enforce_json_file=False, bytes_per_file=0)
        await self.run_test(enforce_json_file=False, bytes_per_file=int(1E12))
        await self.run_test(enforce_json_file=True, bytes_per_file=0)
        await self.run_test(enforce_json_file=True, bytes_per_file=int(1E12))
