"""Interactive sample alignment with explicit cancellation and camera cleanup."""
import asyncio

from PyQt6.QtCore import Qt
from pyqtgraph import ImageView

from concert.base import Parameterizable
from concert.quantities import q


class Alignment(Parameterizable):
    async def __ainit__(self, x_motor, y_motor, z_motor, tomo_motor, camera, pixel_size,
                        tomo_pos_x=0 * q.deg, tomo_pos_y=90 * q.deg,
                        target_horizontal=None, target_vertical=None):
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

    def mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = self._viewer.getImageItem().mapFromScene(event.scenePos())
        elif event.button() == Qt.MouseButton.RightButton:
            self._last_pos = None
        else:
            return
        self._input_event.set()

    async def __call__(self):
        """Run both views; stop the sequence if either view is cancelled."""
        if not await self.align_x(z=True):
            return False
        return await self.align_y()

    async def align_x(self, z=False):
        async with self._lock:
            positions = await self._align(self._tomo_pos_x)
            if positions is None:
                return False
            await self._x_motor.move(positions[0])
            if z:
                await self._z_motor.move(positions[1])
            return True

    async def align_y(self, z=False):
        async with self._lock:
            positions = await self._align(self._tomo_pos_y)
            if positions is None:
                return False
            await self._y_motor.move(positions[0])
            if z:
                await self._z_motor.move(positions[1])
            return True

    async def _align(self, pos):
        trigger_source = await self._camera.get_trigger_source()
        was_recording = await self._camera.get_state() == 'recording'
        try:
            if was_recording:
                await self._camera.stop_recording()
            await self._camera.set_trigger_source('SOFTWARE')
            await self._tomo_motor.set_position(pos)
            async with self._camera.recording():
                await self._camera.trigger()
                img = await self._camera.grab()
            self._viewer.setImage(img.T)
            self._last_pos = None
            self._input_event.clear()
            print('Left-click the sample center; right-click or Ctrl+C to cancel.')
            self._viewer.show()
            await self._input_event.wait()
            if self._last_pos is None:
                return None
            delta_x = self._target_horizontal - self._last_pos.x()
            delta_y = self._target_vertical - self._last_pos.y()
            return delta_x * self._pixel_size, -delta_y * self._pixel_size
        finally:
            self._viewer.hide()
            await self._camera.set_trigger_source(trigger_source)
            if was_recording:
                await self._camera.start_recording()
