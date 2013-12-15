"""
Test experiments. Logging is disabled, so just check the directory and log
files creation.
"""
import shutil
import os
import tempfile
import numpy as np
from concert.coroutines import coroutine
from concert.experiments.base import Acquisition, Experiment
from concert.experiments.imaging import Experiment as ImagingExperiment
from concert.tests import TestCase


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
