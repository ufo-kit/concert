"""
Test experiments. Logging is disabled, so just check the directory and log
files creation.
"""
import shutil
import os
from concert.experiments.base import Experiment
from concert.tests import TestCase


class TestExperiment(TestCase):

    def setUp(self):
        super(TestExperiment, self).setUp()
        self.base_directory = "experiment"

    def tearDown(self):
        shutil.rmtree(self.base_directory)

    def test_single_directory(self):
        experiment = Experiment(lambda: None, self.base_directory)
        experiment.run()
        experiment.run()
        self.assertTrue(os.path.exists(os.path.join(self.base_directory,
                                                    experiment.log_file_name)))
        # There must be no subdirectories
        self.assertEquals(os.listdir(self.base_directory),
                          [experiment.log_file_name])

    def test_multiple_directories(self):
        experiment = Experiment(lambda: None, "experiment/scan{:>03}")
        experiment.run()
        experiment.run()
        directory = os.path.join(self.base_directory,
                                 "scan001", experiment.log_file_name)
        self.assertTrue(os.path.exists(directory))

        directory = os.path.join(self.base_directory,
                                 "scan002", experiment.log_file_name)
        self.assertTrue(os.path.exists(directory))
