import tempfile
import shutil
import numpy as np
import os.path as op
from concert.coroutines.base import async_generate
from concert.storage import DummyWalker, DirectoryWalker, StorageError
from concert.tests import TestCase


class TestWalker(TestCase):

    def setUp(self):
        super(TestWalker, self).setUp()
        self.walker = DummyWalker()
        self.data = [0, 1]

    def check(self, subdir=''):
        truth = set([op.join(subdir, 'foo', str(i)) for i in self.data])
        self.assertTrue(truth.issubset(self.walker.paths))

    async def test_create_writer_no_subdir(self):
        await self.walker.create_writer(async_generate(self.data), dsetname='foo')
        self.check()
        self.assertEqual(self.walker.current, '')

    async def test_create_writer_with_subdir(self):
        await self.walker.create_writer(async_generate(self.data), name='inside', dsetname='foo')
        self.check(subdir='inside')
        self.assertEqual(self.walker.current, '')

    async def test_coroutine(self):
        await self.walker.write(async_generate(self.data), dsetname='foo')
        self.check()

    async def test_lock(self):
        async with self.walker:
            self.assertTrue(self.walker._lock.locked)


class TestDirectoryWalker(TestCase):

    def setUp(self):
        super(TestDirectoryWalker, self).setUp()
        self.path = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self.path)
        self.data = np.ones((2, 2))

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_directory_creation(self):
        self.walker.descend('foo')
        self.walker.descend('bar')
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
        self.walker.descend('foo')
        await self.walker.write(async_generate([self.data]))
        self.assertTrue(op.exists(op.join(self.path, 'foo', 'frame_000000.tif')))

    async def test_custom_write(self):
        await self.walker.write(async_generate([self.data]), dsetname='foo-{}.tif')
        self.assertTrue(op.exists(op.join(self.path, 'foo-0.tif')))

    def test_invalid_ascend(self):
        with self.assertRaises(StorageError):
            self.walker.ascend()

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
