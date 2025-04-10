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


def generate_frames(number=10, dimensions=(10, 10), metadata=True) -> [ImageWithMetadata]:
    frames = []
    for i in range(number):
        image = np.random.random(dimensions)
        image = image.astype(np.float32)
        if metadata:
            metadata = {
                "time": datetime.now().isoformat(),
                "random_value": np.random.random(),
                "shape": dimensions,
                "mirror": False,
                "rotate": 0
            }
            image = ImageWithMetadata(image, metadata=metadata)
        frames.append(image)
    return frames


class TestReaderWriter(TestCase):
    async def asyncSetUp(self) -> None:
        self._data_dir = tempfile.mkdtemp()
        self.data = generate_frames(20)

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir)

    async def run_test(
        self,
        enforce_json_file=False,
        bytes_per_file=0,
        walker_kwargs=None,
        metadata=True,
    ):
        if walker_kwargs is None:
            walker_kwargs = {}
        if 'writer' not in walker_kwargs:
            walker_kwargs['writer'] = TiffWriter

        config.ALWAYS_WRITE_JSON_METADATA_FILE = enforce_json_file
        run_file_name = datetime.now().strftime("%y%m%d_%H%M%S.%f")
        folder = os.path.join(self._data_dir, run_file_name)
        os.mkdir(folder)
        if metadata:
            walker = await DirectoryWalker(
                root=folder,
                writer=TiffWriter,
                dsetname="frame_{:>06}.tif",
                bytes_per_file=bytes_per_file
            )
            await walker.write(async_generate(self.data))
        else:
            # This makes sure no metadata is written starting with second image
            import tifffile
            self.data = np.ones((10,) * 3, dtype=np.float32)
            tifffile.imwrite(os.path.join(folder, "frame_000000.tif"), self.data)
        reader = TiffSequenceReader(os.path.join(folder))
        data2 = []
        num_frames = reader.num_images
        for i in range(num_frames):
            data2.append(reader.read(index=i))
        self.compare_frames(self.data, data2, metadata=metadata)

    def compare_frames(
        self,
        frames_01: [ImageWithMetadata],
        frames_02: [ImageWithMetadata],
        metadata=True
    ) -> None:
        self.assertEqual(len(frames_01), len(frames_02))
        for i in range(len(frames_01)):
            self.assertTrue(np.array_equal(frames_01[i], frames_02[i]))
            if metadata:
                self.assertEqual(
                    frames_01[i].metadata["time"],
                    frames_02[i].metadata["time"]
                )
                self.assertEqual(
                    frames_01[i].metadata["random_value"],
                    frames_02[i].metadata["random_value"]
                )

    async def test_tifffile(self) -> None:
        await self.run_test(enforce_json_file=False, bytes_per_file=0)
        await self.run_test(enforce_json_file=False, bytes_per_file=int(1E12))
        await self.run_test(enforce_json_file=True, bytes_per_file=0)
        await self.run_test(enforce_json_file=True, bytes_per_file=int(1E12))

    async def test_no_metadata(self) -> None:
        # Reader must be able to deal with images without metadata
        await self.run_test(metadata=False, bytes_per_file=1E12)

    async def test_remote_walker_tifffile(self) -> None:
        pass
