import asyncio
import os
import shutil
import tempfile
from typing import Tuple, List
import logging
from time import time
import unittest
import unittest.mock as mock
import numpy as np
import concert
from concert.storage import DirectoryWalker
from concert.experiments.base import Experiment as BaseExperiment, Acquisition, local
from concert.experiments.base import Consumer as AcquisitionConsumer
from concert.tests import TestCase as BaseTestCase, slow
from concert.directors.dummy import Director
from concert.directors.base import Director as BaseDirector
from concert.directors.scanning import XYScan
from concert.devices.motors.dummy import LinearMotor
from concert.quantities import q
from concert.storage import RemoteDirectoryWalker
from concert.tests.util.mocks import MockWalkerDevice


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

    @local
    async def _frame_producer(self):
        yield np.random.random((100, 100))


class BrokenExperiment(Experiment):
    """
    Experiment, that causes an exception within the acquisition call.
    """
    @local
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

    @local
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
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self._data_dir = tempfile.mkdtemp()
        self.walker = await DirectoryWalker(root=self._data_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir)


@slow
class DirectorTest(TestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.experiment = await Experiment(walker=self.walker, separate_scans=False)
        self.director = await Director(experiment=self.experiment, num_iterations=5)
        await self.director.run()

    async def test_final_state(self):
        self.assertEqual(await self.director.get_state(), "standby")
        self.assertEqual(await self.experiment.get_iteration(), 5)


@slow
class DirectorTestBrokenExperiment(TestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
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
        await super().asyncSetUp()
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
        await super().asyncSetUp()
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
        await super().asyncSetUp()
        self.experiment = await Experiment(walker=self.walker, separate_scans=False)
        self.x_motor = await LinearMotor()
        self.y_motor = await LinearMotor()
        self.director = await XYScan(experiment=self.experiment,
                                     x_motor=self.x_motor,
                                     y_motor=self.y_motor,
                                     x_min=0 * q.mm,
                                     x_max=10 * q.mm,
                                     x_step=2.5 * q.mm,
                                     y_min=0 * q.mm,
                                     y_max=10 * q.mm,
                                     y_step=2.5 * q.mm)
        await self.director.run()

    async def test_final_state(self):
        self.assertEqual(await self.director.get_state(), "standby")


class TestableLoggingDirector(BaseDirector):
    """Defines testable director to repeat a given number of experiments, so that some reasonable
    assertions can be made on the logging behavior"""

    _num_iter: int
    _iter_name: str

    async def __ainit__(self, experiment: Experiment, num_iter: int, iter_name: str) -> None:
        self._num_iter = num_iter
        self._iter_name = iter_name
        await super().__ainit__(experiment=experiment)

    async def _get_number_of_iterations(self) -> int:
        return self._num_iter

    async def _prepare_run(self, iteration: int) -> None:
        self.log.info(f"Preparing iteration: {iteration}")

    async def _get_iteration_name(self, iteration: int) -> str:
        return f"{self._iter_name}_{iteration:04d}"

    async def get_iteration_name(self, iteration: int) -> str:
        return await self._get_iteration_name(iteration)

    async def get_iteration(self) -> int:
        return self._get_current_iteration()


class TestDirectorLogging(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        logging.disable(logging.NOTSET)
        self._visited = 0
        self._acquired = 0
        self._root = "root"
        self._director_iter = 3
        self._device = MockWalkerDevice()
        self._walker = await RemoteDirectoryWalker(device=self._device, root=self._root)
        foo = await Acquisition("foo", self.produce, acquire=self.acquire)
        foo.add_consumer(AcquisitionConsumer(self.consume)), Tuple
        bar = await Acquisition("bar", self.produce, acquire=self.acquire)
        self._acquisitions = [foo, bar]
        self.num_produce = 2
        self._item = None
        self._experiment = await BaseExperiment(acquisitions=self._acquisitions,
                                                walker=self._walker)
        await self._experiment._set_log_level("debug")
        self._direxp = await TestableLoggingDirector(experiment=self._experiment,
                                                     num_iter=self._director_iter, iter_name="iter")
        await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        logging.disable(logging.CRITICAL)
        await super().asyncTearDown()

    async def acquire(self):
        self._acquired += 1

    @local
    async def produce(self):
        self._visited += 1
        for i in range(self.num_produce):
            yield np.ones((1,)) * i

    @local
    async def consume(self, producer):
        async for item in producer:
            self._item = item

    async def test_director_logging(self) -> None:
        _ = await self._direxp.run()
        mock_device = self._walker.device.mock_device
        # director.log + (self._director_iter * experiment.log)
        expected_register_call_count = 1 + self._director_iter
        self.assertEqual(mock_device.register_logger.call_count, expected_register_call_count)
        mock_device.register_logger.assert_has_calls([
            mock.call((TestableLoggingDirector.__name__, str(logging.NOTSET), "director.log")),
            mock.call(("Experiment", str(logging.NOTSET), "experiment.log")),
            mock.call(("Experiment", str(logging.NOTSET), "experiment.log")),
            mock.call(("Experiment", str(logging.NOTSET), "experiment.log"))
        ])
        # Expected logs for testable logging director is computed as following
        # Director logging its info_table to root director.log
        root_director_log = 1
        # TestableLoggingDirector calls INFO log from _prepare_run for each iteration
        director_prep_log = self._director_iter
        # Director uses experiment's logger to log its own info_table for each iteration
        director_exp_info_log = self._director_iter
        # Experiment _run logs its own info_table for each iteration
        exp_info_log = self._director_iter
        # Experiment _run triggers one DEBUG log for current iteration start, one DEBUG log for each
        # acquisition, one DEBUG log for acquisition consume finished, one DEBUG log for current
        # iteration duration, all of which happens for each iteration of director
        exp_debug_log = self._director_iter * (1 + len(self._acquisitions) + 1 + 1)
        director_log_total_call_count = (root_director_log + director_prep_log)
        director_log_total_call_count += (director_exp_info_log + exp_info_log + exp_debug_log)
        self.assertEqual(mock_device.log.call_count, director_log_total_call_count)
        expected_deregister_call_count = expected_register_call_count
        log_base_paths: List[str] = [await self._walker.get_current()] * self._director_iter
        director_iterations: List[str] = [await self._direxp.get_iteration_name(
            iteration) for iteration in range(self._director_iter)]
        expected_log_paths = [os.path.join(base_path, iteration,
                                           "experiment.log") for base_path, iteration in zip(
                                               log_base_paths, director_iterations)]
        expected_log_paths.append(os.path.join(await self._walker.get_current(), "director.log"))
        expected_deregister_calls: List[mock.call] = list(map(lambda path: mock.call(path),
                                                              expected_log_paths))
        mock_device.deregister_logger.assert_has_calls(expected_deregister_calls)
        self.assertEqual(mock_device.deregister_logger.call_count, expected_deregister_call_count)
        # JSON logging takes place twice for each iteration
        expected_json_logging_call_count = self._director_iter * 2
        self.assertTrue(mock_device.log_to_json.call_count == expected_json_logging_call_count)
