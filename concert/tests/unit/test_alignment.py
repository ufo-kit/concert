import logging
from types import SimpleNamespace
from typing import Union, Type
import unittest
from unittest.mock import AsyncMock, Mock, call, patch
import numpy as np
from skimage.draw import disk
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.processes import alignment as alg
from concert.quantities import q


def motor(motor_type: Union[Type[LinearMotor], Type[RotationMotor]] = LinearMotor) -> Mock:
    # A method-list spec avoids evaluating Concert's parameter descriptors on the class.
    device = Mock(spec=['move', 'get_position', 'set_position'])
    device.__class__ = motor_type
    device.move = AsyncMock()
    device.set_position = AsyncMock()
    device.get_position = AsyncMock(
        return_value=0 * (q.deg if motor_type is RotationMotor else q.mm))
    return device


def contexts():
    acquisition = alg.AcquisitionContext(
        alg.AcquisitionDevices(AsyncMock(), AsyncMock(), motor(RotationMotor), motor(), motor()),
        height=240, width=400, flat_field_correct=False, absorptivity=False,
    )
    alignment = alg.AlignmentContext(
        alg.AlignmentDevices(motor(RotationMotor), motor(RotationMotor), motor(), motor()),
        pixel_size_um=10 * q.um,
    )
    state = alg.AlignmentState({}, {}, {}, sphere_radius=5)
    return acquisition, alignment, state


def sphere_frame(value=1):
    frame = np.zeros((240, 240), dtype=float)
    rows, columns = disk((120, 120), 8, shape=frame.shape)
    frame[rows, columns] = value
    return frame


class TestMeasurements(unittest.IsolatedAsyncioTestCase):

    async def test_opposing_views_and_projection_offsets(self):
        acq, align, state = contexts()
        template = np.random.default_rng(4).random((9, 9))
        ref, mov = np.zeros((80, 100)), np.zeros((80, 100))
        ref[20:29, 30:39] = template
        mov[26:35, 44:53] = template
        state.patches = {str(angle): template.copy() for angle in (0, 90, 180, 270)}
        state.baseline_scores = {str(angle): 1.0 for angle in (0, 90, 180, 270)}
        for angle in (0, 90):
            with self.subTest(angle=angle):
                acq.devices.camera.grab.side_effect = [ref, mov]
                shifts = await alg.get_sample_shifts(acq, align, state, angle * q.deg)
                self.assertEqual(shifts, (6, 7))
                self.assertTrue(state.sample_in_FOV(ref, angle))
                self.assertFalse(state.sample_in_FOV(np.zeros_like(ref), angle))
        self.assertEqual(acq.devices.tomo_motor.move.await_args_list,
                         [call(180 * q.deg), call(-180 * q.deg)] * 2)
        acq.devices.camera.grab.side_effect = [ref]
        self.assertEqual(await alg.offset_from_projection_center(acq, state), (16, 16))

    async def test_fov_score_rule_includes_absolute_current_score(self):
        _, _, state = contexts()
        state.patches = {"0": np.ones((3, 3))}
        state.baseline_scores = {"0": 0.9}
        with patch.object(alg.sft, 'match_template', return_value=np.array([[-0.8]])):
            self.assertTrue(state.sample_in_FOV(np.zeros((5, 5)), 0))
        state.score_epsilon = 0.1
        with patch.object(alg.sft, 'match_template', return_value=np.array([[0.5]])):
            self.assertFalse(state.sample_in_FOV(np.zeros((5, 5)), 0))

    async def test_crop_default_and_explicit_endpoints(self):
        acq, _, state = contexts()
        frame = np.arange(30).reshape(6, 5)
        acq.devices.camera.grab.return_value = frame
        np.testing.assert_array_equal(await alg.acquire_frame(acq, state), frame)
        for start, end in ((1, 4), (0, -1), (-3, None)):
            with self.subTest(start=start, end=end):
                acq.vert_crop_start, acq.vert_crop_end = start, end
                np.testing.assert_array_equal(await alg.acquire_frame(acq, state), frame[start:end])

    async def test_initialization_without_viewer(self):
        acq, align, _ = contexts()
        frames = [sphere_frame(value) for value in (1, 2, 3, 4)]
        acq.devices.camera.grab.side_effect = frames
        align.proc_func = lambda frame: frame > 0
        with patch.object(alg, 'PyQtGraphViewer') as viewer_constructor:
            state = await alg.init_alignment_state(acq, align)
        viewer_constructor.assert_not_called()
        self.assertEqual(tuple(state.patches), ('0', '90', '180', '270'))
        self.assertEqual(acq.devices.tomo_motor.set_position.await_args_list,
                         [call(angle * q.deg) for angle in (0, 90, 180, 270, 0)])

    async def test_initialization_with_one_mosaic(self):
        acq, align, _ = contexts()
        acq.devices.camera.grab.side_effect = [sphere_frame(v) for v in (1, 2, 3, 4)]
        align.proc_func = lambda frame: frame > 0
        align.viewer = SimpleNamespace(show=AsyncMock(), set_title=AsyncMock())
        with patch.object(alg, 'PyQtGraphViewer') as viewer_constructor:
            state = await alg.init_alignment_state(acq, align)
        viewer_constructor.assert_not_called()
        align.viewer.show.assert_awaited_once()
        align.viewer.set_title.assert_awaited_once()
        self.assertIn('0° | 90° / 180° | 270°', align.viewer.set_title.await_args.args[0])
        preview = align.viewer.show.await_args.args[0]
        self.assertEqual(preview.shape, (400, 400))
        for angle, row, col in (('0', 0, 0), ('90', 0, 200),
                                ('180', 200, 0), ('270', 200, 200)):
            np.testing.assert_array_equal(preview[row:row + 200, col:col + 200],
                                          state.patches[angle])
            self.assertFalse(np.shares_memory(preview, state.patches[angle]))


class TestCorrections(unittest.IsolatedAsyncioTestCase):
    async def test_centering_residuals_and_motor_sequence(self):
        scenarios = (
            ('immediate', [1, 1], [], 0, True),
            ('equality', [2, 2], [], 0, True),
            ('converged', [5, 6, 2, 1], [0.1 * q.mm, -0.1 * q.mm, -50 * q.um], 1, True),
            ('exhausted', [5, 4, 3, 1], [0.1 * q.mm, -0.1 * q.mm, 50 * q.um], 1, False),
        )
        for name, errors, distances, iterations, converged in scenarios:
            with self.subTest(name=name):
                acq, align, state = contexts()
                align.max_iterations = 1
                align.move = AsyncMock()
                logger = logging.getLogger('alignment-test.custom')
                original_debug = logger.debug
                with patch.object(alg, 'get_sample_shifts', new=AsyncMock(
                        side_effect=[(0, error) for error in errors])) as measure:
                    with self.assertLogs(logger, level='INFO') as logs:
                        result = await alg.center_sample_on_axis(acq, align, state, logger)
                self.assertIsNone(result)
                self.assertEqual(logger.debug, original_debug)
                self.assertEqual(align.move.await_args_list,
                                 [call(motor=align.devices.align_motor_obd, distance=d)
                                  for d in distances])
                self.assertEqual(measure.await_count, len(errors))
                self.assertIn('Sample centering OBD', logs.output[0])
                self.assertIn('pixel', logs.output[0])
                self.assertIn('tolerance=2.0 pixel', logs.output[0])
                self.assertIn(f'iterations={iterations}', logs.output[0])
                self.assertIn(f'converged={converged}', logs.output[0])
                warnings = [record for record in logs.records if record.levelno == logging.WARNING]
                self.assertEqual(len(warnings), int(not converged))
                self.assertIn('Sample centering PBD', logs.output[-1])

    async def test_sample_centering_divisor_and_pbd_direction(self):
        acq, align, state = contexts()
        align.adjust_move = 2
        align.move = AsyncMock()
        with patch.object(alg, 'get_sample_shifts', new=AsyncMock(
                side_effect=[(0, 1), (0, 8), (0, 7), (0, 2)])) as measure:
            await alg.center_sample_on_axis(acq, align, state)
        self.assertEqual(align.move.await_args_list, [
            call(motor=align.devices.align_motor_pbd, distance=0.1 * q.mm),
            call(motor=align.devices.align_motor_pbd, distance=-0.1 * q.mm),
            call(motor=align.devices.align_motor_pbd, distance=40 * q.um),
        ])
        self.assertEqual([c.kwargs['tomo_angle'] for c in measure.await_args_list],
                         [0 * q.deg, 90 * q.deg, 90 * q.deg, 90 * q.deg])

    async def test_projection_centering_sequence_and_default_logging(self):
        acq, align, state = contexts()
        align.adjust_move = 2  # Projection centering must still use the full correction.
        acq.move = AsyncMock()
        with patch.object(alg, 'offset_from_projection_center', new=AsyncMock(
                side_effect=[(5, 8), (6, 8), (2, 8), (2, 8), (2, 7), (2, 2)])) as measure:
            with self.assertLogs(alg.LOG, level='INFO') as logs:
                await alg.center_axis_in_projection(acq, align, state)
        self.assertEqual(acq.move.await_args_list, [
            call(motor=acq.devices.z_motor, distance=0.1 * q.mm),
            call(motor=acq.devices.z_motor, distance=-0.1 * q.mm),
            call(motor=acq.devices.z_motor, distance=-50 * q.um),
            call(motor=acq.devices.flat_motor, distance=0.1 * q.mm),
            call(motor=acq.devices.flat_motor, distance=-0.1 * q.mm),
            call(motor=acq.devices.flat_motor, distance=80 * q.um),
        ])
        self.assertEqual(measure.await_count, 6)
        self.assertEqual(len(logs.records), 2)
        for record in logs.records:
            self.assertIn('iterations=1, converged=True', record.getMessage())

    async def test_projection_centering_iteration_limit_continues(self):
        acq, align, state = contexts()
        align.max_iterations = 0
        acq.move = AsyncMock()
        with patch.object(alg, 'offset_from_projection_center', new=AsyncMock(
                side_effect=[(5, 8), (5, 8)])):
            with self.assertLogs(alg.LOG, level='INFO') as logs:
                await alg.center_axis_in_projection(acq, align, state)
        acq.move.assert_not_awaited()
        self.assertEqual(sum(record.levelno == logging.WARNING for record in logs.records), 2)

    async def test_angular_corrections_order_direction_and_residuals(self):
        for converged in (True, False):
            with self.subTest(converged=converged):
                acq, align, state = contexts()
                align.max_iterations = 1
                align.preload, acq.preload = AsyncMock(), AsyncMock()
                events = []

                async def move(motor, distance):
                    events.append((motor, distance))

                async def sample_center(*args, **kwargs):
                    events.append('sample-center')

                async def projection_center(*args, **kwargs):
                    events.append('projection-center')

                align.move = AsyncMock(side_effect=move)
                # Feed scripted angles through the real vertical-separation estimator.
                final = 0 if converged else 1
                angles_and_radii = [(1, 200), (2, 200), (final, 200),
                                    (1.5, 180), (0.5, 180), (final, 180)]
                shifts = [(2 * radius * np.tan(np.deg2rad(angle)), 0)
                          for angle, radius in angles_and_radii]
                with patch.object(alg, 'center_sample_on_axis', new=AsyncMock(
                        side_effect=sample_center)), patch.object(
                        alg, 'center_axis_in_projection', new=AsyncMock(
                            side_effect=projection_center)), patch.object(
                        alg, 'get_sample_shifts', new=AsyncMock(side_effect=shifts)) as measure:
                    with self.assertLogs(alg.LOG, level='INFO') as logs:
                        result = await alg.align_tomo_stage_parallel_beam(acq, align, state)
                self.assertIsNone(result)
                self.assertEqual(events[:2], ['sample-center', 'projection-center'])
                self.assertEqual(events[-1], 'projection-center')
                moves = events[2:-1]
                expected_motors = [align.devices.align_motor_pbd,
                                   align.devices.rot_motor_pitch, align.devices.rot_motor_pitch,
                                   align.devices.rot_motor_pitch, align.devices.align_motor_pbd,
                                   align.devices.align_motor_obd,
                                   align.devices.rot_motor_roll, align.devices.rot_motor_roll,
                                   align.devices.rot_motor_roll, align.devices.align_motor_obd]
                self.assertEqual([event[0] for event in moves], expected_motors)
                expected_distances = [2 * q.mm, 0.05 * q.deg, -0.05 * q.deg, -1 * q.deg,
                                      -2 * q.mm, 1800 * q.um, 0.05 * q.deg, -0.05 * q.deg,
                                      1.5 * q.deg, -1800 * q.um]
                for (_, distance), expected in zip(moves, expected_distances):
                    self.assertAlmostEqual((distance / expected).to_base_units().magnitude, 1)
                self.assertEqual(measure.await_count, 6)
                acq.preload.assert_has_awaits([
                    call(acq.devices.flat_motor), call(acq.devices.z_motor)])
                align.preload.assert_has_awaits([
                    call(align.devices.align_motor_obd), call(align.devices.align_motor_pbd),
                    call(align.devices.rot_motor_roll), call(align.devices.rot_motor_pitch)])
                summaries = [r.getMessage() for r in logs.records if 'residual=' in r.getMessage()]
                self.assertEqual(len(summaries), 2)
                for summary, stage in zip(summaries, ('Pitch correction', 'Roll correction')):
                    self.assertIn(stage, summary)
                    self.assertIn('degree', summary)
                    self.assertIn(f'iterations=1, converged={converged}', summary)
                self.assertEqual(sum(r.levelno == logging.WARNING for r in logs.records),
                                 0 if converged else 2)
                self.assertIn('procedure finished', logs.output[-1])

    async def test_backlash_compensation_movement_sequence(self):
        ctx = alg.BacklashCompRelMovMixin()
        for motor_type, unit in ((LinearMotor, q.mm), (RotationMotor, q.deg)):
            with self.subTest(motor_type=motor_type):
                device = motor(motor_type)
                await ctx.preload(device)
                await ctx.move(device, 0.3 * unit)
                await ctx.move(device, -0.3 * unit)
                self.assertEqual(device.move.await_args_list,
                                 [call(-0.2 * unit), call(0.2 * unit), call(0.3 * unit),
                                  call(-0.4 * unit), call(0.1 * unit)])
