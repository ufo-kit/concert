"""Cartesian scan directors and shared axis configuration."""

import asyncio
from itertools import product
from unittest.mock import AsyncMock, patch

from concert.base import StateError
from concert.devices.motors.dummy import LinearMotor
from concert.directors.scanning import Scan
from concert.experiments.base import Experiment
from concert.quantities import q
from concert.scanning import ScanAxis
from concert.storage import DummyWalker
from concert.tests import TestCase, assert_almost_equal


class PositionExperiment(Experiment):
    async def __ainit__(self, motors, **kwargs):
        self.motors = motors
        self.positions = []
        await super().__ainit__(acquisitions=[], **kwargs)

    async def acquire(self):
        self.positions.append(tuple([await motor.get_position() for motor in self.motors]))


class TestScanDirector(TestCase):
    async def asyncSetUp(self):
        self.motors = [await LinearMotor(), await LinearMotor()]
        self.walker = await DummyWalker()
        self.experiment = await PositionExperiment(self.motors, walker=self.walker)

    async def test_named_axis_mapping(self):
        axes = {'y': (self.motors[1]['position'], [3, 1] * q.mm),
                'x': (self.motors[0]['position'], [2, 5] * q.mm)}
        director = await Scan(axes=axes, experiment=self.experiment, order='snake', go_back=True)
        self.assertEqual(list(director.axes), ['y', 'x'])
        self.assertEqual(await director.get_iteration_name(2), 'iteration_0001_0001')
        await director.run()
        expected = [[2, 3], [5, 3], [5, 1], [2, 1]]
        for actual, values in zip(self.experiment.positions, expected):
            assert_almost_equal(actual, values * q.mm)
        positional = await Scan(axes, experiment=self.experiment)
        self.assertIs(await positional.axes['x'].get_parameter(), self.motors[0]['position'])
        with self.assertRaisesRegex(ValueError, 'not both'):
            await Scan(axes=axes, names=['x', 'y'], experiment=self.experiment)

    async def test_cartesian_scan(self):
        director = await Scan([motor['position'] for motor in self.motors],
                              [[1, 2] * q.mm, [3, 4, 5] * q.mm],
                              experiment=self.experiment, names=['x', 'y'], go_back=True)
        self.assertEqual(await director.get_number_of_iterations(), 6)
        self.assertEqual(await director.get_iteration_name(4), 'iteration_0001_0001')
        self.assertIsInstance(director.axes['x'], ScanAxis)
        await director.run()
        self.assertEqual(len(self.experiment.positions), 6)
        for actual, expected in zip(self.experiment.positions,
                                    product([1, 2] * q.mm, [3, 4, 5] * q.mm)):
            assert_almost_equal(actual, expected)
        for motor in self.motors:
            assert_almost_equal(await motor.get_position(), 0 * q.mm)
        self.assertTrue(await self.experiment.get_separate_runs())

    async def test_single_axis_edit_and_repeat(self):
        director = await Scan(self.motors[0]['position'], [1, 2] * q.mm,
                              experiment=self.experiment, names=['x'])
        await director.axes['x'].set_range(0 * q.mm, 4 * q.mm, 2 * q.mm)
        self.assertIn('position', await director.get_scan_plan())
        self.assertEqual(await director.get_number_of_iterations(), 3)
        await director.run()
        assert_almost_equal([position[0] for position in self.experiment.positions],
                            [0, 2, 4] * q.mm)
        assert_almost_equal(await self.motors[0].get_position(), 4 * q.mm)
        await director.axes['x'].set_values([1, 3] * q.mm)
        await director.run()
        assert_almost_equal([position[0] for position in self.experiment.positions[-2:]],
                            [1, 3] * q.mm)

    async def test_experiment_failure_restores(self):
        director = await Scan(self.motors[0]['position'], [2, 3] * q.mm,
                              experiment=self.experiment, go_back=True)
        with patch.object(self.experiment, 'acquire', new_callable=AsyncMock,
                          side_effect=RuntimeError('experiment failed')):
            with self.assertRaisesRegex(StateError, 'experiment failed'):
                await director.run()
        assert_almost_equal(await self.motors[0].get_position(), 0 * q.mm)
        self.assertTrue(await self.experiment.get_separate_runs())

    async def test_cancellation_and_paused_edit_guard(self):
        director = await Scan(self.motors[0]['position'], [2, 3] * q.mm,
                              experiment=self.experiment, go_back=True)
        entered = asyncio.Event()

        async def acquire():
            entered.set()
            await asyncio.Event().wait()

        with patch.object(self.experiment, 'acquire', side_effect=acquire):
            task = director.run()
            await asyncio.wait_for(entered.wait(), timeout=5)
            with self.assertRaises(StateError):
                await director.axes['axis_0'].set_values([5] * q.mm)
            await director.pause()
            self.assertEqual(await director.get_state(), 'paused')
            with self.assertRaises(StateError):
                await director.axes['axis_0'].set_range(0 * q.mm, 5 * q.mm, 1 * q.mm)
            with self.assertRaises(StateError):
                await director.set_go_back(False)
            with self.assertRaises(StateError):
                await director.set_order('snake')
            with self.assertRaises(StateError):
                await director.set_scan_axis_0_values([5] * q.mm)
            task.cancel()
            # The base director treats cancellation as a normal stop.
            await task
        assert_almost_equal(await self.motors[0].get_position(), 0 * q.mm)
        self.assertTrue(await self.experiment.get_separate_runs())

    async def test_validation(self):
        for parameters, values in [([], []), ([self.motors[0]['position']], []),
                                   (self.motors[0]['position'], [])]:
            with self.assertRaises(ValueError):
                await Scan(parameters, values, experiment=self.experiment)

    async def test_orders_with_arbitrary_lists(self):
        x_values = [8 * q.mm, -1 * q.mm, 2 * q.mm]
        y_values = [3 * q.mm, 7 * q.mm, 1 * q.mm]
        paths = {
            'snake': [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)],
            'spiral': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1)],
        }
        for order, indices in paths.items():
            with self.subTest(order=order):
                self.experiment.positions.clear()
                director = await Scan([motor['position'] for motor in self.motors],
                                      [x_values, y_values], experiment=self.experiment,
                                      names=['x', 'y'], order=order, go_back=True)
                self.assertIn(f'Order: {order}', await director.get_scan_plan())
                for n, (x, y) in enumerate(indices):
                    self.assertEqual(await director.get_iteration_name(n), f'iteration_{x:04d}_{y:04d}')
                await director.run()
                self.assertEqual(len(self.experiment.positions), 9)
                for actual, (x, y) in zip(self.experiment.positions, indices):
                    assert_almost_equal(actual, (x_values[x], y_values[y]))
                for motor in self.motors:
                    assert_almost_equal(await motor.get_position(), 0 * q.mm)

    async def test_order_edit_and_validation(self):
        director = await Scan(self.motors[0]['position'], [3, 1, 2] * q.mm,
                              experiment=self.experiment)
        await director.set_order('snake')
        with self.assertRaises(ValueError):
            await director.set_order('spiral')
        self.assertEqual(await director.get_order(), 'snake')
        await director.run()
        assert_almost_equal([position[0] for position in self.experiment.positions], [3, 1, 2] * q.mm)

    async def test_dynamic_axis_parameters(self):
        director = await Scan(self.motors[0]['position'], [3, 1, 2] * q.mm,
                              experiment=self.experiment, names=['x'])
        self.assertIn('scan_x', director)
        self.assertIn('position: values=[3 millimeter, 1 millimeter, 2 millimeter]',
                      await director.get_scan_x())
        await director.set_scan_x_values([4, 2] * q.mm)
        assert_almost_equal(await director.axes['x'].get_values(), [4, 2] * q.mm)
        self.assertEqual(await director.get_number_of_iterations(), 2)
        self.assertIn('scan_x_values', str(await director.info_table))
        self.assertIn('scan_x', await director._prepare_metadata_str())
        await director.run()
        assert_almost_equal([position[0] for position in self.experiment.positions], [4, 2] * q.mm)

    async def test_early_readiness(self):
        director = await Scan(self.motors[0]['position'], [1, 2] * q.mm,
                              experiment=self.experiment)
        prepared = asyncio.Event()
        original_prepare = director._prepare_run

        async def prepare(iteration):
            await original_prepare(iteration)
            if iteration == 1:
                prepared.set()

        original_acquire = self.experiment.acquire

        async def acquire():
            await original_acquire()
            if len(self.experiment.positions) == 1:
                self.experiment.ready_to_prepare_next_outer_loop.set()
                await asyncio.wait_for(prepared.wait(), timeout=5)

        with patch.object(director, '_prepare_run', side_effect=prepare), \
                patch.object(self.experiment, 'acquire', side_effect=acquire):
            await director.run()
        assert_almost_equal([position[0] for position in self.experiment.positions], [1, 2] * q.mm)
