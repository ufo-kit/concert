"""Image scan acquisition and editable scan plans."""

import asyncio
import json
from itertools import product
from unittest.mock import AsyncMock, patch

import numpy as np

from concert.base import StateError
from concert.devices.cameras.dummy import Camera
from concert.devices.motors.dummy import LinearMotor
from concert.experiments.addons.local import Accumulator
from concert.experiments.base import Consumer, ConsumerError, local, remote
from concert.experiments.scan import LocalScan, RemoteScan, Scan
from concert.quantities import q
from concert.helpers import ImageWithMetadata
from concert.tests import TestCase, assert_almost_equal


class TestScanExperiment(TestCase):
    async def asyncSetUp(self):
        self.motor = await LinearMotor()
        self.camera = await Camera(background=np.ones((2, 3)), simulate=False)

    async def test_cartesian_acquisition(self):
        experiment = await Scan(
            [self.motor['position'], self.camera['exposure_time']],
            [[0, 1] * q.mm, [1, 2, 3] * q.ms],
            camera=self.camera, names=['x', 'exposure'], go_back=True)
        accumulator = await Accumulator(experiment=experiment)
        positions = []
        original_trigger = self.camera.trigger

        async def trigger():
            positions.append((await self.motor.get_position(),
                              await self.camera.get_exposure_time()))
            await original_trigger()

        with patch.object(self.camera, 'trigger', side_effect=trigger):
            await experiment.run()
        images = await accumulator.get_items(experiment.scan)
        self.assertEqual(len(images), 6)
        for image in images:
            np.testing.assert_array_equal(image, np.ones((2, 3)))
        for actual, expected in zip(positions, product([0, 1] * q.mm, [1, 2, 3] * q.ms)):
            assert_almost_equal(actual, expected)
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
        assert_almost_equal(await self.camera.get_exposure_time(), 1 * q.ms)
        self.assertEqual(await self.camera.get_trigger_source(), 'AUTO')
        self.assertEqual(await self.camera.get_state(), 'standby')

    async def test_single_axis_and_adjustment(self):
        experiment = await Scan(self.motor['position'], [0, 1] * q.mm,
                                camera=self.camera, names=['x'])
        axis = experiment.axes['x']
        self.assertIs(await axis.get_parameter(), self.motor['position'])
        await axis.set_range(0 * q.mm, 1 * q.cm, 5 * q.mm)
        assert_almost_equal(await axis.get_values(), [0, 5, 10] * q.mm)
        self.assertEqual(await experiment.get_num_positions(), 3)
        plan = await experiment.get_scan_plan()
        self.assertIn('position', plan)
        self.assertIn('x', plan)
        await experiment.run()
        assert_almost_equal(await self.motor.get_position(), 10 * q.mm)
        await axis.set_values([2, 3] * q.mm)
        self.assertIsNone(await axis.get_start())
        await experiment.run()
        assert_almost_equal(await self.motor.get_position(), 3 * q.mm)

    async def test_descending_and_exclusive_ranges(self):
        experiment = await Scan(self.motor['position'], [0] * q.mm, camera=self.camera)
        axis = experiment.axes['axis_0']
        await axis.set_range(3 * q.mm, 0 * q.mm, -2 * q.mm)
        assert_almost_equal(await axis.get_values(), [3, 1, 0] * q.mm)
        await axis.set_range(0 * q.mm, 3 * q.mm, 2 * q.mm, include_last=False)
        assert_almost_equal(await axis.get_values(), [0, 2] * q.mm)
        for step in [0, -1]:
            with self.assertRaises(ValueError):
                await axis.set_range(0 * q.mm, 3 * q.mm, step * q.mm)
        assert_almost_equal(await axis.get_values(), [0, 2] * q.mm)

    async def test_failure_cleanup(self):
        experiment = await Scan(self.motor['position'], [2, 3] * q.mm,
                                camera=self.camera, go_back=True)
        with patch.object(self.camera, 'grab', side_effect=RuntimeError('grab failed')):
            with self.assertRaisesRegex(RuntimeError, 'grab failed'):
                await experiment.run()
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
        self.assertEqual(await self.camera.get_trigger_source(), 'AUTO')
        self.assertEqual(await self.camera.get_state(), 'standby')

    async def test_consumer_failure_cleanup(self):
        experiment = await Scan(self.motor['position'], [2, 3] * q.mm,
                                camera=self.camera, go_back=True)

        @local
        async def fail(producer):
            async for image in producer:
                raise RuntimeError('consumer failed')

        experiment.scan.add_consumer(Consumer(fail))
        with self.assertRaisesRegex(RuntimeError, 'consumer failed'):
            await experiment.run()
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
        self.assertEqual(await self.camera.get_trigger_source(), 'AUTO')
        self.assertEqual(await self.camera.get_state(), 'standby')

    async def test_cancellation_and_edit_guard(self):
        experiment = await Scan(self.motor['position'], [2, 3] * q.mm,
                                camera=self.camera, go_back=True)
        entered = asyncio.Event()

        async def grab():
            entered.set()
            await asyncio.Event().wait()

        with patch.object(self.camera, 'grab', side_effect=grab):
            task = experiment.run()
            await asyncio.wait_for(entered.wait(), timeout=5)
            with self.assertRaises(StateError):
                await experiment.axes['axis_0'].set_values([5] * q.mm)
            with self.assertRaises(StateError):
                await experiment.set_go_back(False)
            with self.assertRaises(StateError):
                await experiment.set_order('snake')
            with self.assertRaises(StateError):
                await experiment['scan_axis_0_values'].set([5] * q.mm)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
        self.assertEqual(await self.camera.get_trigger_source(), 'AUTO')
        self.assertEqual(await self.camera.get_state(), 'standby')

    async def test_validation(self):
        with self.assertRaises(ValueError):
            await Scan([], [], camera=self.camera)
        with self.assertRaises(ValueError):
            await Scan(self.motor['position'], [], camera=self.camera)
        with self.assertRaises(ValueError):
            await Scan([self.motor['position'], self.camera['exposure_time']],
                       [[0] * q.mm], camera=self.camera)
        with self.assertRaises(ValueError):
            await Scan([self.motor['position']] * 2, [[0] * q.mm] * 2, camera=self.camera)
        with self.assertRaises(ValueError):
            await Scan(self.camera['trigger_source'], ['AUTO'], camera=self.camera)
        with self.assertRaises(ValueError):
            await Scan(self.motor['position'], [0] * q.mm, camera=self.camera, names=['', 'x'])

    async def test_explicit_local_scan(self):
        experiment = await LocalScan(self.motor['position'], [0, 1] * q.mm, camera=self.camera)
        self.assertFalse(experiment.scan.remote)
        self.assertTrue(issubclass(Scan, LocalScan))
        accumulator = await Accumulator(experiment=experiment)
        await experiment.run()
        self.assertEqual(len(await accumulator.get_items(experiment.scan)), 2)

    async def test_named_axis_mapping(self):
        for cls in [LocalScan, RemoteScan]:
            for positional in [False, True]:
                with self.subTest(mode=cls.__name__, positional=positional):
                    axes = {
                        'exposure': (self.camera['exposure_time'], [1, 3] * q.ms),
                        'x': (self.motor['position'], [2, 5] * q.mm),
                    }
                    if positional:
                        experiment = await cls(axes, camera=self.camera, order='snake')
                    else:
                        experiment = await cls(axes=axes, camera=self.camera, order='snake')
                    self.assertEqual(list(experiment.axes), ['exposure', 'x'])
                    self.assertIs(await experiment.axes['x'].get_parameter(), self.motor['position'])
                    self.assertIs(await experiment.axes['exposure'].get_parameter(),
                                  self.camera['exposure_time'])
                    self.assertEqual(await experiment.get_num_positions(), 4)
                    self.assertIn('scan_exposure', experiment)
                    with patch.object(self.camera, 'grab_send', new_callable=AsyncMock):
                        await experiment.run()
                    assert_almost_equal(await self.motor.get_position(), 2 * q.mm)
                    assert_almost_equal(await self.camera.get_exposure_time(), 3 * q.ms)

    async def test_axis_mapping_validation(self):
        valid = {'x': (self.motor['position'], [1] * q.mm)}
        for kwargs in [dict(axes={}), dict(axes=[]), dict(axes={'x': self.motor['position']}),
                       dict(axes={'x': (self.motor['position'],)}), dict(axes=valid, names=['x']),
                       dict(axes=valid, values=[1] * q.mm), dict(parameters=valid, axes=valid),
                       dict(axes={'x': ('position', [1] * q.mm)}),
                       dict(axes={'x': (self.motor['position'], [])}),
                       dict(axes={'': (self.motor['position'], [1] * q.mm)})]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    await LocalScan(camera=self.camera, **kwargs)
        with self.assertRaises(ValueError):
            await LocalScan(axes={'trigger': (self.camera['trigger_source'], ['AUTO'])},
                            camera=self.camera)

    async def test_dynamic_axis_parameters(self):
        for cls in [LocalScan, RemoteScan]:
            with self.subTest(mode=cls.__name__):
                experiment = await cls(self.motor['position'], [5, 1, 3] * q.mm,
                                       camera=self.camera, names=['x'])
                self.assertIn('scan_x', experiment)
                self.assertFalse(experiment['scan_x'].writable)
                self.assertTrue(experiment['scan_x_values'].writable)
                self.assertEqual(await experiment.get_scan_x_num_points(), 3)
                self.assertIn('values=[5 millimeter, 1 millimeter, 3 millimeter]',
                              await experiment.get_scan_x())
                await experiment.set_scan_x_values([9, 2] * q.mm)
                assert_almost_equal(await experiment.axes['x'].get_values(), [9, 2] * q.mm)
                await experiment.axes['x'].set_range(0 * q.mm, 4 * q.mm, 2 * q.mm)
                assert_almost_equal(await experiment.get_scan_x_start(), 0 * q.mm)
                assert_almost_equal(await experiment.get_scan_x_stop(), 4 * q.mm)
                assert_almost_equal(await experiment.get_scan_x_step(), 2 * q.mm)
                self.assertEqual(await experiment.get_scan_x_num_points(), 3)
                self.assertIn('start=0 millimeter, stop=4 millimeter, step=2 millimeter',
                              await experiment.get_scan_x())
                await experiment.set_order('snake')
                self.assertIn('order=snake', await experiment.get_scan_x())
                self.assertIn('scan_x', str(await experiment.info_table))
                self.assertIn('scan_x', await experiment._prepare_metadata_str())

    async def test_local_image_scan_metadata(self):
        experiment = await LocalScan(
            [self.motor['position'], self.camera['exposure_time']],
            [[2, 5] * q.mm, [1, 3] * q.ms], camera=self.camera,
            names=['x', 'exposure'], order='snake', go_back=True)
        accumulator = await Accumulator(experiment=experiment)
        # The same camera image can be reused; never mutate its metadata.
        camera_image = ImageWithMetadata(np.ones((2, 3)), metadata={
            'mirror': True, 'rotate': 1, 'conversion_applied': True,
            'camera': {'serial': 'test-camera'},
        })
        with patch.object(self.camera, 'grab', new_callable=AsyncMock, return_value=camera_image):
            await experiment.run()
        images = await accumulator.get_items(experiment.scan)
        self.assertEqual(len(images), 4)
        expected = [(0, 0, '2 millimeter', '0.001 second'),
                    (0, 1, '2 millimeter', '0.003 second'),
                    (1, 1, '5 millimeter', '0.003 second'),
                    (1, 0, '5 millimeter', '0.001 second')]
        for n, (image, (x, exposure, position_value, exposure_value)) in enumerate(zip(images, expected)):
            np.testing.assert_array_equal(image, camera_image)
            self.assertEqual(image.metadata['camera'], {'serial': 'test-camera'})
            self.assertTrue(image.metadata['conversion_applied'])
            self.assertEqual(image.metadata['scan'], {
                'iteration': n, 'indices': [x, exposure], 'order': 'snake',
                'axes': {'x': {'parameter': 'position', 'value': position_value},
                         'exposure': {'parameter': 'exposure_time', 'value': exposure_value}},
            })
            json.dumps(image.metadata)
        self.assertNotIn('scan', camera_image.metadata)
        images[0].metadata['camera']['serial'] = 'changed'
        self.assertEqual(images[1].metadata['camera']['serial'], 'test-camera')
        self.assertEqual(camera_image.metadata['camera']['serial'], 'test-camera')
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)

    async def test_local_metadata_uses_readback_before_trigger(self):
        experiment = await LocalScan(self.motor['position'], [2] * q.mm,
                                     camera=self.camera)
        original_trigger = self.camera.trigger

        async def trigger():
            await original_trigger()
            await self.motor.set_position(3 * q.mm)

        # Requested position is 2, read-back is 2.5 before the trigger;
        # neither the requested value nor a post-trigger read should be stored.
        original_get = self.motor['position'].get

        async def get_position():
            return await original_get() + 0.5 * q.mm

        with patch.object(self.motor['position'], 'get', side_effect=get_position), \
                patch.object(self.camera, 'trigger', side_effect=trigger):
            images = [image async for image in experiment.take_scan()]
        self.assertEqual(images[0].metadata['scan']['axes']['axis_0'],
                         {'parameter': 'position', 'value': '2.5 millimeter'})

    async def test_dynamic_parameters_are_instance_specific(self):
        first = await LocalScan(self.motor['position'], [1] * q.mm,
                                camera=self.camera, names=['x'])
        second = await LocalScan(self.motor['position'], [2] * q.mm,
                                 camera=self.camera, names=['y'])
        self.assertNotIn('scan_y', first)
        self.assertNotIn('scan_x', second)
        self.assertFalse(hasattr(first, 'get_scan_y'))
        self.assertFalse(hasattr(second, 'get_scan_x'))
        self.assertIsInstance(first, LocalScan)
        self.assertIsInstance(second, LocalScan)

    async def test_dynamic_parameter_name_sanitization(self):
        experiment = await LocalScan(self.motor['position'], [1] * q.mm,
                                     camera=self.camera, names=['sample x'])
        self.assertEqual(experiment.axis_parameter_names['sample x']['values'], 'scan_sample_x_values')
        self.assertIn('scan_sample_x', experiment)
        with self.assertRaisesRegex(ValueError, 'conflicting parameter'):
            await LocalScan([self.motor['position'], self.camera['exposure_time']],
                            [[1] * q.mm, [1] * q.ms], camera=self.camera,
                            names=['x', 'x_values'])

    async def test_remote_scan(self):
        experiment = await RemoteScan(
            [self.motor['position'], self.camera['exposure_time']],
            [[2, 3] * q.mm, [1, 2] * q.ms],
            camera=self.camera, names=['x', 'exposure'], go_back=True)
        self.assertTrue(experiment.scan.remote)
        self.assertIs(experiment.scan.producer, self.camera)
        positions = []

        async def send(num, end=True):
            self.assertEqual(num, 1)
            positions.append((await self.motor.get_position(),
                              await self.camera.get_exposure_time(), end))

        with patch.object(self.camera, 'grab_send', new_callable=AsyncMock, side_effect=send) as sender, \
                patch.object(self.camera, 'grab', side_effect=AssertionError('local grab')):
            await experiment.run()
            self.assertEqual(sender.await_count, 4)
        for actual, expected in zip(positions, product([2, 3] * q.mm, [1, 2] * q.ms)):
            assert_almost_equal(actual[:2], expected)
        self.assertEqual([position[2] for position in positions], [False, False, False, True])
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
        self.assertEqual(await self.camera.get_trigger_source(), 'AUTO')
        self.assertEqual(await self.camera.get_state(), 'standby')

    async def test_remote_count_and_adjustment(self):
        experiment = await RemoteScan(self.motor['position'], [0] * q.mm,
                                      camera=self.camera, names=['x'])
        await experiment.axes['x'].set_range(0 * q.mm, 2 * q.mm, 1 * q.mm)
        with patch.object(self.camera, 'grab_send', new_callable=AsyncMock) as sender:
            self.assertEqual(await experiment.take_scan(), 3)
            self.assertEqual(sender.await_count, 3)
            sender.assert_awaited_with(1, end=True)

    async def test_remote_failure_cleanup(self):
        experiment = await RemoteScan(self.motor['position'], [2, 3] * q.mm,
                                      camera=self.camera, go_back=True)
        with patch.object(self.camera, 'grab_send', side_effect=RuntimeError('send failed')):
            with self.assertRaisesRegex(RuntimeError, 'send failed'):
                await experiment.run()
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
        self.assertEqual(await self.camera.get_trigger_source(), 'AUTO')
        self.assertEqual(await self.camera.get_state(), 'standby')

    async def test_remote_cancellation_cleanup(self):
        experiment = await RemoteScan(self.motor['position'], [2, 3] * q.mm,
                                      camera=self.camera, go_back=True)
        entered = asyncio.Event()

        async def send(num, end=True):
            entered.set()
            await asyncio.Event().wait()

        with patch.object(self.camera, 'grab_send', side_effect=send):
            task = experiment.run()
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
        self.assertEqual(await self.camera.get_trigger_source(), 'AUTO')
        self.assertEqual(await self.camera.get_state(), 'standby')

    async def test_orders_and_arbitrary_lists_local_and_remote(self):
        values = [[8 * q.mm, -1 * q.mm, 2 * q.mm], [3 * q.ms, 7 * q.ms, 1 * q.ms]]
        paths = {
            'snake': [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)],
            'spiral': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1)],
        }
        for cls in [LocalScan, RemoteScan]:
            for order, indices in paths.items():
                with self.subTest(mode=cls.__name__, order=order):
                    experiment = await cls([self.motor['position'], self.camera['exposure_time']],
                                           values, camera=self.camera, names=['x', 'exposure'],
                                           order=order, go_back=True)
                    positions = []
                    original_trigger = self.camera.trigger

                    async def trigger():
                        positions.append((await self.motor.get_position(),
                                          await self.camera.get_exposure_time()))
                        await original_trigger()

                    with patch.object(self.camera, 'trigger', side_effect=trigger), \
                            patch.object(self.camera, 'grab_send', new_callable=AsyncMock) as sender:
                        await experiment.run()
                    self.assertEqual(len(positions), 9)
                    for actual, (x, y) in zip(positions, indices):
                        assert_almost_equal(actual, (values[0][x], values[1][y]))
                    if cls is RemoteScan:
                        self.assertEqual([call.kwargs['end'] for call in sender.await_args_list],
                                         [False] * 8 + [True])
                    assert_almost_equal(await self.motor.get_position(), 0 * q.mm)
                    self.assertIn(f'Order: {order}', await experiment.get_scan_plan())
                    await experiment.axes['x'].set_values([5 * q.mm, 1 * q.mm])
                    await experiment.set_order('raster')
                    self.assertEqual(await experiment.get_num_positions(), 6)

    async def test_invalid_order(self):
        with self.assertRaises(ValueError):
            await LocalScan(self.motor['position'], [0] * q.mm, camera=self.camera, order='spiral')
        experiment = await LocalScan(self.motor['position'], [0] * q.mm, camera=self.camera)
        with self.assertRaises(ValueError):
            await experiment.set_order('spiral')
        self.assertEqual(await experiment.get_order(), 'raster')

    async def test_consumer_modes(self):
        local_scan = await LocalScan(self.motor['position'], [0] * q.mm, camera=self.camera)
        remote_scan = await RemoteScan(self.motor['position'], [0] * q.mm, camera=self.camera)

        @local
        async def local_consumer(producer):
            pass

        @remote
        async def remote_consumer():
            pass

        with self.assertRaises(ConsumerError):
            local_scan.scan.add_consumer(Consumer(remote_consumer))
        with self.assertRaises(ConsumerError):
            remote_scan.scan.add_consumer(Consumer(local_consumer))
        remote_scan.scan.add_consumer(Consumer(remote_consumer))
        with patch.object(self.camera, 'grab_send', new_callable=AsyncMock):
            await remote_scan.run()
