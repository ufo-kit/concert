import tempfile
import shutil
import numpy as np
import os.path as op
import tifffile
from concert.coroutines.base import async_generate
from concert.storage import DummyWalker, DirectoryWalker, RemoteDirectoryWalker
from concert.storage import StorageError
from concert.tests import TestCase
from concert.tests.util.mocks import MockWalkerDevice


class TestWalker(TestCase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.walker = await DummyWalker()
        self.data = [0, 1]

    async def check(self, subdir=''):
        truth = set([op.join(subdir, 'foo', str(i)) for i in self.data])
        self.assertTrue(truth.issubset(await self.walker.paths))

    async def test_create_writer_no_subdir(self):
        await self.walker.create_writer(async_generate(self.data), dsetname='foo')
        await self.check()
        self.assertEqual(await self.walker.get_current(), '')

    async def test_create_writer_with_subdir(self):
        await self.walker.create_writer(async_generate(self.data), name='inside', dsetname='foo')
        await self.check(subdir='inside')
        self.assertEqual(await self.walker.get_current(), '')

    async def test_coroutine(self):
        print(f"Self Data: {self.data}")
        await self.walker.write(async_generate(self.data), dsetname='foo')
        await self.check()

    async def test_lock(self):
        async with self.walker:
            self.assertTrue(self.walker._lock.locked)


class TestDirectoryWalker(TestCase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.path = tempfile.mkdtemp()
        self.walker = await DirectoryWalker(root=self.path)
        self.data = np.ones((2, 2))

    def tearDown(self):
        shutil.rmtree(self.path)

    async def test_directory_creation(self):
        await self.walker.descend('foo')
        await self.walker.descend('bar')
        self.assertTrue(op.exists(op.join(self.path, 'foo')))
        self.assertTrue(op.exists(op.join(self.path, 'foo', 'bar')))

    async def test_default_write(self):
        await self.walker.write(async_generate([self.data, self.data]))
        self.assertTrue(op.exists(op.join(self.path, 'frame_000000.tif')))
        self.assertTrue(op.exists(op.join(self.path, 'frame_000001.tif')))

        # Cannot write if directory is not empty
        with self.assertRaises(StorageError):
            await self.walker.write(async_generate([self.data]))

        # Make a new one ...
        await self.walker.descend('foo')
        await self.walker.write(async_generate([self.data]))
        self.assertTrue(op.exists(op.join(self.path, 'foo', 'frame_000000.tif')))

    async def test_custom_write(self):
        await self.walker.write(async_generate([self.data]), dsetname='foo-{}.tif')
        self.assertTrue(op.exists(op.join(self.path, 'foo-0.tif')))

    async def test_write_image(self):
        image = np.arange(12, dtype=np.uint16).reshape(3, 4)
        await self.walker.write_image(image, 'single.tif')

        path = op.join(self.path, 'single.tif')
        self.assertTrue(op.exists(path))
        np.testing.assert_array_equal(tifffile.imread(path), image)

        await self.walker.descend('inside')
        await self.walker.write_image(image, 'nested.tif')
        np.testing.assert_array_equal(
            tifffile.imread(op.join(self.path, 'inside', 'nested.tif')),
            image
        )

    async def test_invalid_ascend(self):
        with self.assertRaises(StorageError):
            await self.walker.ascend()

    async def test_dset_exists(self):
        await self.walker.write(async_generate([self.data]))
        with self.assertRaises(StorageError):
            await self.walker.write(async_generate([self.data]))

    async def test_same_directory_different_dset(self):
        await self.walker.write(async_generate([self.data]))
        await self.walker.write(async_generate([self.data]), dsetname='bar-{}.tif')

    async def test_dset_prefix(self):
        async def test_raises(dsetname):
            with self.assertRaises(ValueError):
                await self.walker.write(async_generate([self.data]), dsetname=dsetname)

        async def test_ok(dsetname):
            await self.walker.write(async_generate([self.data]), dsetname=dsetname)

        await test_ok('bar-{}.tif')
        await test_ok('baz-{:>06}.tif')
        await test_ok('spam-{0}')

        await test_raises('bar')
        await test_raises('bar-{')
        await test_raises('bar-}')
        await test_raises('bar-}{')
        await test_raises('bar-}{{}')


class TestRemoteDirectoryWalker(TestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.device = MockWalkerDevice()
        self.walker = await RemoteDirectoryWalker(device=self.device)

    async def test_write_image(self):
        image = np.arange(24, dtype=np.uint16).reshape(3, 4, 2)

        await self.walker.write_image(image, "remote.tif")

        encoding, blob = self.device.mock_device.write_image.await_args.args[0]
        self.assertEqual(encoding, "remote.tif:4:3:2:uint16")
        np.testing.assert_array_equal(
            np.frombuffer(blob, dtype=np.uint16).reshape(3, 4, 2),
            image
        )
