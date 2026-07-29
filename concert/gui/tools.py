import asyncio
from asyncio import Event

from pyqtgraph import ImageView
from concert.coroutines.base import background
from concert.quantities import q
from concert.base import Parameterizable
from qasync import asyncSlot
from concert.coroutines.base import run_in_loop

class Alignment(Parameterizable):
    async def __ainit__(self, x_motor, y_motor, z_motor, tomo_motor, camera, pixel_size, tomo_pos_x=0*q.deg, tomo_pos_y=90*q.deg, target_horizontal=None, target_vertical=None):
        """
        Align the sample to the camera
        :param x_motor:
        :param y_motor:
        :param tomo_motor:
        :param tomo_pos_x:
        :param tomo_pos_y:
        :param pixel_size:
        :param target:
        :return:
        """
        self._x_motor = x_motor
        self._y_motor = y_motor
        self._z_motor = z_motor
        self._tomo_motor = tomo_motor
        self._tomo_pos_x = tomo_pos_x
        self._tomo_pos_y = tomo_pos_y
        self._pixel_size = pixel_size
        self._camera = camera
        if target_horizontal is None:
            self._target_horizontal = (await self._camera.get_roi_width()).magnitude/ 2
        if target_vertical is None:
            self._target_vertical = (await self._camera.get_roi_height()).magnitude / 2
        await super().__ainit__()
        self._viewer = ImageView()
        self._input_event = Event()
        self._viewer.scene.sigMouseClicked.connect(self.mouse_clicked)
        #self._viewer.scene.keyPressEvent.connect(self.keyboard_pressed)
        self._last_pos = None

    def mouse_clicked(self, mouseClickEvent):
        pos = self._viewer.getImageItem().mapFromScene(mouseClickEvent.scenePos())
        if mouseClickEvent.button() == 1:
            self._last_pos = pos
        elif mouseClickEvent.button() == 2:
            self._last_pos = None
        self._last_pos = pos
        self._input_event.set()

    def __call__(self, *args, **kwargs):
        run_in_loop(self.align_x(z=True))
        run_in_loop(asyncio.sleep(1))
        run_in_loop(self.align_y(z=False))

    async def align_x(self, z=False):
        positions = await self._align(self._tomo_pos_x)
        if positions is None:
            return
        await self._x_motor.move(positions[0])
        if z:
            await self._z_motor.move(positions[1])

    async def align_y(self, z=False):
        positions = await self._align(self._tomo_pos_y)
        if positions is None:
            return
        await self._y_motor.move(positions[0])
        if z:
            await self._z_motor.move(positions[1])

    async def _align(self, pos):
        if await self._camera.get_state() == "recording":
            await self._camera.stop_recording()
        await self._camera.set_trigger_source("SOFTWARE")

        await self._tomo_motor.set_position(pos)

        async with self._camera.recording():
            await self._camera.trigger()
            img = await self._camera.grab()

        self._viewer.setImage(img.T)
        print("Click on the center of the sample. Ctrl+C to cancel.")
        self._viewer.show()
        self._input_event.clear()
        await self._input_event.wait()
        self._viewer.hide()
        if self._last_pos is None:
            return
        delta_x = self._target_horizontal - self._last_pos.x()
        delta_y = self._target_vertical - self._last_pos.y()
        return delta_x * self._pixel_size, -delta_y*self._pixel_size
