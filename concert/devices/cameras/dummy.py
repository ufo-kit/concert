"""This module provides a simple dummy camera."""
import asyncio
import time
import numpy as np
from concert.base import Parameter
from concert.coroutines.base import run_in_executor
from concert.quantities import q
from concert.base import check, transition, Quantity
from concert.devices.cameras import base
from concert.devices.motors.dummy import TomographyStage
from concert.readers import TiffSequenceReader


class Base(base.Camera):

    exposure_time = Quantity(q.s)
    sensor_pixel_width = Quantity(q.micrometer)
    sensor_pixel_height = Quantity(q.micrometer)
    roi_x0 = Quantity(q.pixel)
    roi_y0 = Quantity(q.pixel)
    roi_width = Quantity(q.pixel)
    roi_height = Quantity(q.pixel)

    async def __ainit__(self):
        await super(Base, self).__ainit__()
        self._frame_rate = 1000 / q.s
        self._trigger_source = self.trigger_sources.AUTO
        self._exposure_time = 1 * q.ms
        self._roi_x0 = 0 * q.pixel
        self._roi_y0 = 0 * q.pixel
        self._roi_width = 640 * q.pixel
        self._roi_height = 480 * q.pixel

    async def _get_sensor_pixel_width(self):
        return 5 * q.micrometer

    async def _get_sensor_pixel_height(self):
        return 5 * q.micrometer

    async def _get_roi_x0(self):
        return self._roi_x0

    async def _set_roi_x0(self, x0):
        self._roi_x0 = x0

    async def _get_roi_y0(self):
        return self._roi_y0

    async def _set_roi_y0(self, y0):
        self._roi_y0 = y0

    async def _get_roi_width(self):
        return self._roi_width

    async def _set_roi_width(self, roi):
        self._roi_width = roi

    async def _get_roi_height(self):
        return self._roi_height

    async def _set_roi_height(self, roi):
        self._roi_height = roi

    async def _get_exposure_time(self):
        return self._exposure_time

    async def _set_exposure_time(self, value):
        self._exposure_time = value

    async def _get_frame_rate(self):
        return self._frame_rate

    async def _set_frame_rate(self, frame_rate):
        self._frame_rate = frame_rate

    async def _get_trigger_source(self):
        return self._trigger_source

    async def _set_trigger_source(self, source):
        self._trigger_source = source

    @transition(target='recording')
    async def _record_real(self):
        pass

    @transition(target='standby')
    async def _stop_real(self):
        pass

    async def _trigger_real(self):
        pass


class Camera(Base):

    """A simple dummy camera."""

    simulate = Parameter(help='Simulate noise')

    async def __ainit__(self, background=None, simulate=True):
        """
        *background* can be an array-like that will be used to generate the frame when calling grab.
        If *simulate* is True the final image intensity will be scaled based on exposure time and
        poisson noise will be added. If *simulate* is False, the background will be returned with no
        modifications to it.
        """
        await super(Camera, self).__ainit__()
        self._simulate = simulate
        await self.set_background(background)

    async def _get_simulate(self):
        return self._simulate

    async def _set_simulate(self, value):
        self._simulate = value

    async def set_background(self, background):
        if background is not None:
            self._roi_width = background.shape[1] * q.pixel
            self._roi_height = background.shape[0] * q.pixel
            self._background = background
        else:
            shape = (640, 480)
            self._roi_width, self._roi_height = shape * q.pixel
            self._background = np.ones(shape[::-1], dtype=np.uint16)

    async def _grab_real(self):
        if not await self.get_simulate():
            return self._background
        start = time.time()
        cur_time = (await self.get_exposure_time()).to(q.s).magnitude
        # 1e5 is a dummy correlation between exposure time and emitted e-.
        tmp = self._background + cur_time * 1e5
        max_value = np.iinfo(np.uint16).max
        tmp = np.random.poisson(tmp)
        # Cut values beyond the bit-depth.
        tmp[tmp > max_value] = max_value
        duration = time.time() - start
        to_sleep = 1 / (await self.get_frame_rate())
        to_sleep = to_sleep.to_base_units() - duration * q.s
        if to_sleep > 0 * q.s:
            await asyncio.sleep(to_sleep.magnitude)

        return np.asarray(tmp, dtype=np.uint16)


class FileCamera(Base):

    """A camera that reads files specified by *pattern*. It can be a directory, in which case all
    the files inside are read, or it can be a pattern and only the matching files will be read. If
    *reset_on_start* is True the files are read from the beginning when the recording starts.
    *start_index* specifies the index of the first read image (not file index, in case the files are
    multi-page).
    """

    pattern = Parameter(help='Image file pattern to read')

    async def __ainit__(self, pattern, reset_on_start=True, start_index=0):
        # Let users change the directory
        await super(FileCamera, self).__ainit__()
        self._pattern = pattern
        self._reader = TiffSequenceReader(pattern)
        self._start_index = start_index
        self.index = start_index
        self.reset_on_start = reset_on_start
        image = self._reader.read(0)
        self._roi_height = image.shape[0] * q.px
        self._roi_width = image.shape[1] * q.px

    async def _get_pattern(self):
        return self._pattern

    @check(source='standby')
    async def _set_pattern(self, pattern):
        self._pattern = pattern
        self._reader = TiffSequenceReader(pattern)
        self.index = self._start_index

    @transition(target='recording')
    async def _record_real(self):
        if self.reset_on_start:
            self.index = 0

    async def _grab_real(self):
        roi_x0 = await self.get_roi_x0()
        roi_y0 = await self.get_roi_y0()
        if await self.get_roi_height():
            y_region = (roi_y0 + await self.get_roi_height()).magnitude
        else:
            y_region = None
        if await self.get_roi_width():
            x_region = (roi_x0 + await self.get_roi_width()).magnitude
        else:
            x_region = None

        image = await run_in_executor(self._reader.read, self.index)
        self.index += 1
        await asyncio.sleep((1 / (await self.get_frame_rate())).to(q.s).magnitude)

        return image[roi_y0.magnitude:y_region, roi_x0.magnitude:x_region]


class BufferedCamera(Camera, base.BufferedMixin):

    async def __ainit__(self, background=None, simulate=True):
        await Camera.__ainit__(self, background=background, simulate=simulate)

    @transition(target='readout')
    async def _start_readout_real(self):
        pass

    @transition(target='standby')
    async def _stop_readout_real(self):
        pass

    async def _readout_real(self):
        for i in range(3):
            yield await self.grab()


class TomographyStageCamera(Base):

    """A dummy camera which places an object on the tomography stack."""

    async def __ainit__(self, shape=(512, 512), pixel_size=None, sphere_radius=None):
        await super().__ainit__()
        self.stage = await TomographyStage()
        self._shape = shape
        self._pixel_size = 1 * q.um if pixel_size is None else pixel_size
        self._sphere_radius = sphere_radius if sphere_radius else int(min(shape) / 30)

    async def get_center(self):
        """Sample is by default located at (0, 0, 0), this computes the new position based on motor
        positions.
        """
        from concert.geometry import rotate, translate, X_AX, Y_AX, Z_AX

        def _vectorize(value, position):
            shift = [0, 0, 0]
            shift[position] = value.magnitude

            return shift * value.units

        parallel_below = _vectorize(await self.stage.vertical_motor_below.get_actual_position(), 2)
        ortho_below = _vectorize(await self.stage.orthogonal_motor_below.get_actual_position(), 0)
        parallel_above = _vectorize(await self.stage.parallel_motor_above.get_actual_position(), 1)
        ortho_above = _vectorize(await self.stage.orthogonal_motor_above.get_actual_position(), 0)
        lamino_angle = await self.stage.lamino_motor.get_actual_position()
        roll_angle = await self.stage.roll_motor.get_actual_position()
        tomo_angle = await self.stage.tomo_motor.get_actual_position()

        mat_lamino = rotate(lamino_angle, X_AX, shift=-self.stage.lamino_motor_z_offset)
        mat_roll = rotate(roll_angle, Y_AX, shift=-self.stage.roll_motor_z_offset)
        mat_tomo = rotate(tomo_angle, Z_AX)
        mat_parallel_above = translate(parallel_above)
        mat_parallel_below = translate(parallel_below)
        mat_ortho_above = translate(ortho_above)
        mat_ortho_below = translate(ortho_below)

        mat = np.dot(mat_parallel_below, mat_ortho_below)
        mat = np.dot(mat, mat_lamino)
        mat = np.dot(mat, mat_roll)
        mat = np.dot(mat, mat_tomo)
        mat = np.dot(mat, mat_parallel_above)
        mat = np.dot(mat, mat_ortho_above)

        # The nominal center is (0, 0, 0) -> multiplying the transformation matrix with this
        # is equivalent to taking the last matrix column (stripped of the last homogeneous
        # coordinate)
        return mat[:-1, -1] * 1e3 * q.mm
        
    async def _grab_real(self):
        from concert.imageprocessing import make_sphere

        x, _, y = (await self.get_center() / self._pixel_size).to_base_units().magnitude  # in pixels
        x += self._shape[1] // 2
        y += self._shape[0] // 2

        return make_sphere(self._shape, self._sphere_radius, (y, x), fast=True)
