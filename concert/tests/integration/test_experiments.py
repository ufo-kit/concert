"""
Test experiments. Logging is disabled, so just check the directory and log
files creation.
"""
import shutil
import os
import tempfile
import numpy as np
from concert.coroutines import coroutine
from concert.quantities import q
from concert.experiments.base import Acquisition, Experiment
from concert.experiments.imaging import (Experiment as ImagingExperiment,
                                         tomo_angular_step, tomo_max_speed,
                                         tomo_projections_number)
from concert.tests import TestCase, suppressed_logging, assert_almost_equal


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


class TestExperimentBase(TestCase):

    def setUp(self):
        super(TestExperimentBase, self).setUp()
        self.base_directory = tempfile.mkdtemp()
        self.item = None

    def tearDown(self):
        shutil.rmtree(self.base_directory)

    @coroutine
    def consume(self):
        while True:
            self.item = yield


class TestExperiment(TestExperimentBase):

    def setUp(self):
        super(TestExperiment, self).setUp()
        self.item = None
        self.start = 0
        self.foo = Acquisition("foo", self.produce, consumer_callers=[self.consume])
        self.bar = Acquisition("bar", self.produce, consumer_callers=[self.consume])
        self.acquisitions = [self.foo, self.bar]
        self.experiment = Experiment(self.acquisitions, self.base_directory)

    def produce(self, start=0):
        for i in range(self.start, self.start + 2):
            yield i

    def test_run(self):
        self.experiment.run().join()
        self.assertTrue(os.path.exists(self.experiment.directory))
        self.assertEqual(self.item, 1)

        self.start = 2
        self.experiment.run().join()
        self.assertTrue(os.path.exists(self.experiment.directory))
        self.assertEqual(self.item, 3)

    def test_swap(self):
        self.experiment.swap(self.foo, self.bar)
        self.assertEqual(self.acquisitions[0], self.bar)
        self.assertEqual(self.acquisitions[1], self.foo)


class TestImagingExperiment(TestExperimentBase):

    def setUp(self):
        super(TestImagingExperiment, self).setUp()
        self.foo = Acquisition("foo", self.produce, consumer_callers=[self.consume])
        self.experiment = ImagingExperiment([self.foo], os.path.join(self.base_directory,
                                            "scan_{:>03}"))

    def produce(self):
        for i in range(2):
            yield np.ones((2, 2)) * (i + 1)

    def test_run(self):
        self.experiment.run().join()
        directory = os.path.join(self.experiment.directory, self.foo.name)
        self.assertEqual(sorted(os.listdir(directory)), ['frame_00000.tif', 'frame_00001.tif'])
