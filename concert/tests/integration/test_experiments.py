"""
Test experiments. Logging is disabled, so just check the directory and log
files creation.
"""
import asyncio
import numpy as np
import os.path as op
import tempfile
import shutil
from concert.quantities import q
from concert.coroutines.base import start
from concert.coroutines.sinks import Accumulate
from concert.experiments.base import Acquisition, Experiment, ExperimentError
from concert.experiments.imaging import (tomo_angular_step, tomo_max_speed,
                                         tomo_projections_number, frames)
from concert.experiments.addons import Addon, Consumer, ImageWriter, Accumulator
from concert.devices.cameras.dummy import Camera
from concert.tests import TestCase, suppressed_logging, assert_almost_equal
from concert.storage import DummyWalker, DirectoryWalker


class VisitChecker(object):

    """Use this to check that a callback was called."""

    def __init__(self):
        self.visited = False

    async def visit(self, *args, **kwargs):
        self.visited = True


class DummyAddon(Addon):
    def __init__(self):
        self.attached_num_times = 0
        super(DummyAddon, self).__init__([])

    def _attach(self):
        self.attached_num_times += 1

    def _detach(self):
        self.attached_num_times -= 1


class ExperimentSimple(Experiment):
    async def __ainit__(self, walker):
        acq = await Acquisition("test", self._run_test_acq)
        await super(ExperimentSimple, self).__ainit__([acq], walker)

    async def _run_test_acq(self):
        for i in range(10):
            await asyncio.sleep(0.1)
            yield np.random.random((100, 100))


class ExperimentException(Experiment):
    async def __ainit__(self, walker):
        acq = await Acquisition("test", self._run_test_acq)
        await super(ExperimentException, self).__ainit__([acq], walker)

    async def _run_test_acq(self):
        for i in range(2):
            yield i
        raise Exception("Run test acq")


@suppressed_logging
def test_tomo_angular_step():
    truth = np.arctan(2 / 100) * q.rad
    assert_almost_equal(truth, tomo_angular_step(100 * q.px))


@suppressed_logging
def test_projections_number():
    width = 100
    truth = int(np.ceil(np.pi / np.arctan(2 / width) * q.rad))
    assert truth == tomo_projections_number(width * q.px)


@suppressed_logging
def test_tomo_max_speed():
    width = 100
    frame_rate = 100 / q.s
    truth = np.arctan(2 / width) * q.rad * frame_rate
    assert_almost_equal(truth, tomo_max_speed(width * q.px, frame_rate))


class TestImagingFunctions(TestCase):

    async def test_frames(self):
        acc = Accumulate()
        await acc(frames(5, await Camera()))
        assert len(acc.items) == 5


class TestAcquisition(TestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.acquired = False
        self.item = None
        self.acquisition = await Acquisition('foo', self.produce, consumers=[self.consume],
                                             acquire=self.acquire)

    async def test_run(self):
        await self.acquisition()
        self.assertTrue(self.acquired)
        self.assertEqual(1, self.item)

    async def acquire(self):
        self.acquired = True

    async def produce(self):
        yield 1

    async def consume(self, producer):
        async for item in producer:
            self.item = item


class TestExperimentBase(TestCase):

    async def asyncSetUp(self):
        await super(TestExperimentBase, self).asyncSetUp()
        self.acquired = 0
        self.root = ''
        self.walker = DummyWalker(root=self.root)
        self.name_fmt = 'scan_{:>04}'
        self.visited = 0
        self.foo = await Acquisition("foo", self.produce, consumers=[self.consume],
                                     acquire=self.acquire)
        self.bar = await Acquisition("bar", self.produce, acquire=self.acquire)
        self.acquisitions = [self.foo, self.bar]
        self.num_produce = 2
        self.item = None

    async def acquire(self):
        self.acquired += 1

    async def produce(self):
        self.visited += 1
        for i in range(self.num_produce):
            yield i

    async def consume(self, producer):
        async for item in producer:
            self.item = item


class TestExperiment(TestExperimentBase):

    async def asyncSetUp(self):
        await super(TestExperiment, self).asyncSetUp()
        self.experiment = await Experiment(self.acquisitions, self.walker, name_fmt=self.name_fmt)
        self.visit_checker = VisitChecker()

    async def test_run(self):
        await self.experiment.run()
        self.assertEqual(self.visited, len(self.experiment.acquisitions))
        self.assertEqual(self.acquired, len(self.experiment.acquisitions))

        await self.experiment.run()
        self.assertEqual(self.visited, 2 * len(self.experiment.acquisitions))

        truth = set([op.join(self.root, self.name_fmt.format(i)) for i in range(2)])
        self.assertEqual(truth, self.walker.paths)

        # Consumers must be called
        self.assertTrue(self.item is not None)

    def test_swap(self):
        self.experiment.swap(self.foo, self.bar)
        self.assertEqual(self.experiment.acquisitions[0], self.bar)
        self.assertEqual(self.experiment.acquisitions[1], self.foo)

    def test_get_by_name(self):
        self.assertEqual(self.foo, self.experiment.get_acquisition('foo'))
        self.assertRaises(ExperimentError, self.experiment.get_acquisition, 'non-existing')

    def test_acquisition_access(self):
        with self.assertRaises(AttributeError):
            self.experiment.acquisitions.remove(self.bar)

    def test_add(self):
        self.assertEqual(self.experiment.foo, self.foo)
        self.assertEqual(self.experiment.bar, self.bar)

    def test_remove(self):
        self.experiment.remove(self.bar)
        self.assertFalse(hasattr(self.experiment, 'bar'))
        self.assertNotIn(self.bar, self.experiment.acquisitions)

    async def test_prepare(self):
        self.experiment.prepare = self.visit_checker.visit
        await self.experiment.run()
        self.assertTrue(self.visit_checker.visited)

    async def test_finish(self):
        self.experiment.finish = self.visit_checker.visit
        await self.experiment.run()
        self.assertTrue(self.visit_checker.visited)

    async def test_consumer_addon(self):
        accumulate = Accumulate()
        Consumer([self.acquisitions[0]], accumulate)
        await self.experiment.run()
        self.assertEqual(accumulate.items, list(range(self.num_produce)))

    async def test_image_writing(self):
        ImageWriter(self.acquisitions, self.walker)
        await self.experiment.run()

        scan_name = self.name_fmt.format(0)
        # Check if the writing coroutine has been attached
        for i in range(self.num_produce):
            foo = op.join(self.root, scan_name, 'foo', self.walker.dsetname, str(i))
            bar = op.join(self.root, scan_name, 'bar', self.walker.dsetname, str(i))

            self.assertTrue(self.walker.exists(foo))
            self.assertTrue(self.walker.exists(bar))

    async def test_accumulation(self):
        acc = Accumulator(self.acquisitions)
        await self.experiment.run()

        for acq in self.acquisitions:
            self.assertEqual(acc.items[acq], list(range(self.num_produce)))

        # Test detach
        acc.detach()
        for acq in self.acquisitions:
            for consumer in acq.consumers:
                self.assertFalse(isinstance(consumer, Accumulate))
        self.assertEqual(acc.items, {})

    def test_attach_num_times(self):
        """An attached addon cannot be attached the second time."""
        addon = DummyAddon()
        # Does nothing because it's attached during construction
        addon.attach()
        self.assertEqual(1, addon.attached_num_times)

        # Detach
        addon.detach()
        self.assertEqual(0, addon.attached_num_times)

        # Second time, cannot be called
        addon.detach()
        self.assertEqual(0, addon.attached_num_times)


class TestExperimentStates(TestCase):
    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def setUp(self):
        super(TestExperimentStates, self).setUp()
        self.data_dir = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self.data_dir)

    async def test_experiment_state_normal(self):
        exp = await ExperimentSimple(self.walker)
        self.assertEqual(await exp.get_state(), "standby")
        exp_handle = start(exp.run())
        await asyncio.sleep(0.2)
        self.assertEqual(await exp.get_state(), "running")
        await exp_handle
        self.assertEqual(await exp.get_state(), "standby")
        exp_handle = start(exp.run())
        await asyncio.sleep(0.2)
        self.assertEqual(await exp.get_state(), "running")
        exp_handle.cancel()
        try:
            await exp_handle
        except asyncio.CancelledError:
            self.assertEqual(await exp.get_state(), "cancelled")

    async def test_experiment_exception(self):
        exp = await ExperimentException(self.walker)
        self.assertEqual(await exp.get_state(), "standby")
        with self.assertRaises(Exception):
            await exp.run()
        self.assertEqual(await exp.get_state(), "error")
