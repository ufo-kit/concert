"""
Test experiments. Logging is disabled, so just check the directory and log
files creation.
"""
import numpy as np
import os.path as op
from concert.quantities import q
from concert.coroutines.base import coroutine, inject
from concert.experiments.base import Acquisition, Experiment, ExperimentError
from concert.experiments.imaging import (Experiment as ImagingExperiment,
                                         tomo_angular_step, tomo_max_speed,
                                         tomo_projections_number, frames)
from concert.devices.cameras.dummy import Camera
from concert.tests import TestCase, suppressed_logging, assert_almost_equal, VisitChecker
from concert.storage import DummyWalker


@suppressed_logging
def test_tomo_angular_step():
    truth = np.arctan(2.0 / 100) * q.rad
    assert_almost_equal(truth, tomo_angular_step(100 * q.px))


@suppressed_logging
def test_projections_number():
    width = 100
    truth = int(np.ceil(np.pi / np.arctan(2.0 / width) * q.rad))
    assert truth == tomo_projections_number(width * q.px)


@suppressed_logging
def test_tomo_max_speed():
    width = 100
    frame_rate = 100 / q.s
    truth = np.arctan(2.0 / width) * q.rad * frame_rate
    assert_almost_equal(truth, tomo_max_speed(width * q.px, frame_rate))


@suppressed_logging
def test_frames():
    @coroutine
    def count():
        while True:
            yield
            count.i += 1
    count.i = 0
    inject(frames(5, Camera()), count())
    assert count.i == 5


class TestExperimentBase(TestCase):

    def setUp(self):
        super(TestExperimentBase, self).setUp()
        self.root = ''
        self.walker = DummyWalker(root=self.root)
        self.name_fmt = 'scan_{:>04}'
        self.visited = 0
        self.foo = Acquisition("foo", self.produce, consumer_callers=[self.consume])
        self.bar = Acquisition("bar", self.produce)
        self.acquisitions = [self.foo, self.bar]
        self.num_produce = 2
        self.item = None

    def produce(self):
        self.visited += 1
        for i in range(self.num_produce):
            yield i

    @coroutine
    def consume(self):
        while True:
            self.item = yield


class TestExperiment(TestExperimentBase):

    def setUp(self):
        super(TestExperiment, self).setUp()
        self.experiment = Experiment(self.acquisitions, self.walker, name_fmt=self.name_fmt)
        self.visit_checker = VisitChecker()

    def test_run(self):
        self.experiment.run().join()
        self.assertEqual(self.visited, len(self.experiment.acquisitions))

        self.experiment.run().join()
        self.assertEqual(self.visited, 2 * len(self.experiment.acquisitions))

        truth = set([op.join(self.root, self.name_fmt.format(i + 1)) for i in range(2)])
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

    def test_prepare(self):
        self.experiment.prepare = self.visit_checker.visit
        self.experiment.run().join()
        self.assertTrue(self.visit_checker.visited)

    def test_finish(self):
        self.experiment.finish = self.visit_checker.visit
        self.experiment.run().join()
        self.assertTrue(self.visit_checker.visited)


class TestImagingExperiment(TestExperimentBase):

    def setUp(self):
        super(TestImagingExperiment, self).setUp()
        self.experiment = ImagingExperiment(self.acquisitions, self.walker, self.name_fmt)

    def test_run(self):
        self.experiment.run().join()

        scan_name = self.name_fmt.format(1)
        # Check if the writing coroutine has been attached
        for i in range(self.num_produce):
            foo = op.join(self.root, scan_name, 'foo', self.walker.dsetname, str(i))
            bar = op.join(self.root, scan_name, 'bar', self.walker.dsetname, str(i))

            self.assertTrue(self.walker.exists(foo))
            self.assertTrue(self.walker.exists(bar))
