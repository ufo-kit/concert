import asyncio

import shutil
import tempfile
import numpy as np
import logging
from time import time

import concert
from concert.storage import DirectoryWalker
from concert.experiments.base import Experiment as BaseExperiment, Acquisition
from concert.tests import TestCase as BaseTestCase, slow
from concert.directors.dummy import Director
from concert.directors.base import Director as BaseDirector
from concert.directors.scanning import XYScan
from concert.devices.motors.dummy import LinearMotor
from concert.quantities import q


LOG = logging.getLogger(__name__)
concert.config.PROGRESS_BAR = False


class Experiment(BaseExperiment):
    """
    Simple Experiment for tests, that produces one random image (100 x 100) within a acquisition
    *test*.
    """
    async def __ainit__(self, walker, separate_scans):
        acquisition = await Acquisition("test", self._frame_producer)
        await super().__ainit__(acquisitions=[acquisition], walker=walker,
                                separate_scans=separate_scans)

    async def _frame_producer(self):
        yield np.random.random((100, 100))


class BrokenExperiment(Experiment):
    """
    Experiment, that causes an exception within the acquisition call.
    """
    async def _frame_producer(self):
        yield None
        raise Exception("Experiment broken")


class EarlyReadyExperiment(Experiment):
    """
    An experiment, that sets the ready_to_prepare_next_sample and then waits for two seconds within
    the acquisition.
    """
    async def __ainit__(self, walker, separate_scans, set_ready):
        await super().__ainit__(walker, separate_scans)
        self._set_ready = set_ready
        self.ready_time = {}
        self.acq_finished_time = {}

    async def _frame_producer(self):
        yield np.random.random((100, 100))
        if self._set_ready:
            self.ready_to_prepare_next_sample.set()
        self.ready_time[await self.get_iteration()] = time()
        await asyncio.sleep(2)
        self.acq_finished_time[await self.get_iteration()] = time()


class TimeLoggingDirector(BaseDirector):
    """
    Director with two (identical) iterations.
    The _prepare_run() stores the time when it is called.
    """
    async def __ainit__(self, experiment):
        await super().__ainit__(experiment=experiment)
        self.preparation_time = {}

    async def _get_number_of_iterations(self) -> int:
        return 2

    async def _prepare_run(self, iteration: int):
        self.preparation_time[iteration] = time()


class TestCase(BaseTestCase):
    """
    Test case that sets up a walker, runs self.director and deletes the data in the walker
     at the end.
    """
    def setUp(self):
        self._data_dir = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self._data_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir)


@slow
class DirectorTest(TestCase):
    async def asyncSetUp(self):
        self.experiment = await Experiment(walker=self.walker, separate_scans=False)
        self.director = await Director(experiment=self.experiment, num_iterations=5)
        await self.director.run()

    async def test_final_state(self):
        self.assertEqual(await self.director.get_state(), "standby")
        self.assertEqual(await self.experiment.get_iteration(), 5)


@slow
class DirectorTestBrokenExperiment(TestCase):
    async def asyncSetUp(self):
        self.experiment = await BrokenExperiment(walker=self.walker, separate_scans=False)
        self.director = await Director(experiment=self.experiment, num_iterations=5)

    async def test_final_state(self):
        try:
            # This whill cause an expected exception
            await self.director.run()
        except Exception as e:
            LOG.info(e)
        self.assertEqual(await self.director.get_state(), "error")


@slow
class EarlyFinishExperiment(TestCase):
    async def asyncSetUp(self):
        self.experiment = await EarlyReadyExperiment(walker=self.walker,
                                                     separate_scans=False,
                                                     set_ready=True)
        self.director = await TimeLoggingDirector(experiment=self.experiment)
        await self.director.run()

    async def test_early_prepare_time(self):
        """
        Tests if the time, when the ready_to_prepare_next_sample was set is similar to the time when
        the prepare_next_sample was called.
        """
        self.assertAlmostEqual(self.experiment.ready_time[0],
                               self.director.preparation_time[1],
                               delta=0.2)


@slow
class NotEarlyFinishExperiment(TestCase):
    async def asyncSetUp(self):
        self.experiment = await EarlyReadyExperiment(walker=self.walker,
                                                     separate_scans=False,
                                                     set_ready=False)
        self.director = await TimeLoggingDirector(experiment=self.experiment)
        await self.director.run()

    async def test_prepare_time(self):
        """
        Tests if the time of the preparation of iteration 1 corresponds to the time when the
        acquisition of the run 0 is similar.
        """
        self.assertAlmostEqual(self.experiment.acq_finished_time[0],
                               self.director.preparation_time[1],
                               delta=0.2)


@slow
class XYScanDirectorTest(TestCase):
    async def asyncSetUp(self):
        self.experiment = await Experiment(walker=self.walker, separate_scans=False)
        self.x_motor = await LinearMotor()
        self.y_motor = await LinearMotor()
        self.director = await XYScan(experiment=self.experiment,
                                     x_motor=self.x_motor,
                                     y_motor=self.y_motor,
                                     x_min=0*q.mm,
                                     x_max=10*q.mm,
                                     x_step=2.5*q.mm,
                                     y_min=0*q.mm,
                                     y_max=10*q.mm,
                                     y_step=2.5*q.mm)
        await self.director.run()

    async def test_final_state(self):
        self.assertEqual(await self.director.get_state(), "standby")
