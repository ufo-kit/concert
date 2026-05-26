import shutil
import tempfile
import numpy as np
from concert.experiments.addons import Accumulator

from concert.storage import DirectoryWalker

from concert.quantities import q

from concert.tests import TestCase
from concert.devices.cameras.dummy import Camera as DummyCamera
from concert.devices.motors.dummy import LinearMotor
from concert.experiments.mapping import RectangularMotorMapping as AbstractRectangularMotorMapping


class RectangularMotorMapping(AbstractRectangularMotorMapping):
    async def position_sample(self, position_x, position_y):
        await self.motor_x.set_position(position_x)
        await self.motor_y.set_position(position_y)

    async def __ainit__(self, walker, camera, x_motor, y_motor, x_range, y_range, x_step, y_step):
        await super().__ainit__(walker=walker,
                                camera=camera,
                                x_motor=x_motor,
                                y_motor=y_motor,
                                effective_pixel_size=10 * q.um,
                                field_of_view_x=2 * q.mm,
                                field_of_view_y=3 * q.mm,
                                center_x=0 * q.mm,
                                center_y=0 * q.mm,
                                size_x=1 * q.cm,
                                size_y=1 * q.cm,
                                overlap=0.1,
                                separate_scans=False)
        self.motor_x = x_motor
        self.motor_y = y_motor
        self.beam = "off"

    async def start_sample_exposure(self):
        self.beam = "on"

    async def stop_sample_exposure(self):
        self.beam = "off"


class LoggingCamera(DummyCamera):
    async def __ainit__(self):
        self.exp = None
        self._last_motor_x = None
        self._last_motor_y = None
        self._last_beam = None
        await super().__ainit__()

    async def _trigger_real(self):
        self._last_motor_x = await self.exp.motor_x.get_position()
        self._last_motor_y = await self.exp.motor_y.get_position()
        self._last_beam = self.exp.beam

    async def _grab_real(self):
        from concert.helpers import ImageWithMetadata
        if await self.get_trigger_source() == "AUTO":
            await self.trigger()
        img = np.zeros((10, 10)).view(ImageWithMetadata)
        img.metadata['motor_x'] = self._last_motor_x.to(q.mm).magnitude
        img.metadata['motor_y'] = self._last_motor_y.to(q.mm).magnitude
        img.metadata['beam'] = self._last_beam
        return img


class TestMapping(TestCase):
    async def asyncSetUp(self) -> None:
        self.motor_x = await LinearMotor()
        self.motor_y = await LinearMotor()
        self.camera = await LoggingCamera()
        self._data_dir = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self._data_dir, bytes_per_file=1E12)
        self.experiment = await RectangularMotorMapping(self.walker, self.camera, self.motor_x,
                                                        self.motor_y,
                                                        x_range=(0 * q.mm, 10 * q.mm),
                                                        y_range=(0 * q.mm, 10 * q.mm),
                                                        x_step=1 * q.mm, y_step=1 * q.mm)
        self.camera.exp = self.experiment

    async def asyncTearDown(self) -> None:
        shutil.rmtree(self._data_dir)

    async def test_mapping(self):
        acc = Accumulator(self.experiment.acquisitions)
        await self.experiment.run()
        for dark in acc.items[self.experiment.get_acquisition("darks")]:
            self.assertEqual(dark.metadata['beam'], "off", "Beam should be off for dark images")

        positions = await self.experiment.sample_positions()
        for i, radio in enumerate(acc.items[self.experiment.get_acquisition("mapping")]):
            self.assertEqual(radio.metadata['beam'], "on", "Beam should be on for mapping images")
            self.assertEqual(radio.metadata['motor_x'], (positions[i])[0].to(q.mm).magnitude,
                             "Motor x position should be correct")
            self.assertEqual(radio.metadata['motor_y'], (positions[i])[1].to(q.mm).magnitude,
                             "Motor y position should be correct")

        acc.detach()
