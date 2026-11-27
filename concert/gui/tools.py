"""Interactive sample alignment with explicit cancellation and camera cleanup."""
import asyncio
import logging
import inspect

from PyQt6.QtCore import Qt, QEvent, QObject
from PyQt6.QtWidgets import QMessageBox
from pyqtgraph import ImageView, ScatterPlotItem

from concert.base import AsyncObject
from concert.quantities import q
from concert.gui.controller import validate_quantity


LOG = logging.getLogger(__name__)


class _CloseFilter(QObject):
    def __init__(self, cancel):
        super().__init__()
        self.cancel = cancel

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Close:
            self.cancel()
        return False


class Alignment(AsyncObject):
    async def __ainit__(self, x_motor, y_motor, z_motor, tomo_motor, camera, pixel_size,
                        tomo_pos_x=0 * q.deg, tomo_pos_y=90 * q.deg,
                        target_horizontal=None, target_vertical=None, confirmation=None):
        """Align clicked sample centers with the target pixel coordinates.

        Left-click accepts a target; right-click cancels the current alignment.
        Translation offsets use ``pixel_size`` to convert pixels into motor units.
        """
        self._x_motor = x_motor
        self._y_motor = y_motor
        self._z_motor = z_motor
        self._tomo_motor = tomo_motor
        self._tomo_pos_x = tomo_pos_x
        self._tomo_pos_y = tomo_pos_y
        self._pixel_size = pixel_size
        self._camera = camera
        self._target_horizontal = (target_horizontal if target_horizontal is not None else
                                   (await camera.get_roi_width()).magnitude / 2)
        self._target_vertical = (target_vertical if target_vertical is not None else
                                 (await camera.get_roi_height()).magnitude / 2)
        await super().__ainit__()
        self._viewer = ImageView()
        self._input_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._viewer.scene.sigMouseClicked.connect(self.mouse_clicked)
        self._last_pos = None
        self._confirmation = confirmation or self._confirm
        self._close_filter = _CloseFilter(self.cancel)
        self._viewer.installEventFilter(self._close_filter)
        self._target_marker = ScatterPlotItem(symbol='+', size=20, pen='g')
        self._selected_marker = ScatterPlotItem(symbol='o', size=15, pen='y')
        self._viewer.getView().addItem(self._target_marker)
        self._viewer.getView().addItem(self._selected_marker)

    def cancel(self):
        self._last_pos = None
        self._input_event.set()

    async def _confirm(self, positions):
        # open() avoids a nested Qt event loop inside the running asyncio loop.
        dialog = QMessageBox(self._viewer)
        dialog.setWindowTitle('Confirm alignment movement')
        dialog.setText(
            f'Proposed displacement: horizontal {positions[0]}, vertical {positions[1]}.\n'
            'Apply movement? Rotation remains at the alignment angle on cancellation.')
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        result = asyncio.get_running_loop().create_future()

        def finished(_result):
            if not result.done():
                result.set_result(dialog.standardButton(dialog.clickedButton()) ==
                                  QMessageBox.StandardButton.Yes)

        dialog.finished.connect(finished)
        dialog.open()
        try:
            return await result
        finally:
            dialog.close()
            dialog.deleteLater()

    def mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = self._viewer.getImageItem().mapFromScene(event.scenePos())
            self._selected_marker.setData([self._last_pos.x()], [self._last_pos.y()])
        elif event.button() == Qt.MouseButton.RightButton:
            self._last_pos = None
        else:
            return
        self._input_event.set()

    async def __call__(self):
        """Run both views; stop the sequence if either view is cancelled."""
        async with self._lock:
            if not await self._align_axis(self._x_motor, self._tomo_pos_x, z=True):
                return False
            return await self._align_axis(self._y_motor, self._tomo_pos_y, z=False)

    async def align_x(self, z=False):
        async with self._lock:
            return await self._align_axis(self._x_motor, self._tomo_pos_x, z)

    async def align_y(self, z=False):
        async with self._lock:
            return await self._align_axis(self._y_motor, self._tomo_pos_y, z)

    async def _align_axis(self, motor, angle, z):
        positions = await self._align(angle)
        if positions is None:
            return False
        moves = [(motor, positions[0])] + ([(self._z_motor, positions[1])] if z else [])
        await self._validate_moves(moves)
        completed = 0
        try:
            for axis, delta in moves:
                await axis.move(delta)
                completed += 1
        except BaseException as error:
            message = (f'Alignment movement interrupted after {completed} completed axis moves. '
                       'Motor positions may have changed; no automatic rollback was attempted.')
            LOG.error(message)
            if hasattr(error, 'add_note'):
                error.add_note(message)
            raise
        return True

    async def _validate_moves(self, moves):
        # Validate every axis before moving any translation axis. Device setters
        # still enforce limits at execution time; this is not a hardware interlock.
        for motor, delta in moves:
            await validate_quantity(motor['position'], await motor.get_position() + delta)

    async def _align(self, pos):
        trigger_source = await self._camera.get_trigger_source()
        was_recording = await self._camera.get_state() == 'recording'
        original_error = None
        try:
            if was_recording:
                await self._camera.stop_recording()
            await self._camera.set_trigger_source('SOFTWARE')
            await self._tomo_motor.set_position(pos)
            async with self._camera.recording():
                await self._camera.trigger()
                img = await self._camera.grab()
            self._viewer.setImage(img.T)
            self._target_marker.setData([self._target_horizontal], [self._target_vertical])
            self._selected_marker.setData([], [])
            self._last_pos = None
            self._input_event.clear()
            print('Left-click the sample center; right-click or Ctrl+C to cancel.')
            self._viewer.show()
            await self._input_event.wait()
            if self._last_pos is None:
                return None
            delta_x = self._target_horizontal - self._last_pos.x()
            delta_y = self._target_vertical - self._last_pos.y()
            positions = delta_x * self._pixel_size, -delta_y * self._pixel_size
            accepted = self._confirmation(positions)
            if inspect.isawaitable(accepted):
                accepted = await accepted
            return positions if accepted else None
        except BaseException as error:
            original_error = error
            raise
        finally:
            self._viewer.hide()
            async def restore():
                await self._camera.set_trigger_source(trigger_source)
                if was_recording:
                    await self._camera.start_recording()
            cleanup = asyncio.create_task(restore())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            except Exception as error:
                if original_error is None:
                    raise
                LOG.exception('Camera restoration failed after alignment error')
                if hasattr(original_error, 'add_note'):
                    original_error.add_note(f'Camera restoration also failed: {error}')
