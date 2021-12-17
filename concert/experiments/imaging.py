"""Imaging experiments."""
import asyncio
import numpy as np
from concert.quantities import q
from concert.experiments.base import Experiment, Acquisition
from concert.base import background, Parameter, Quantity


async def frames(num_frames, camera, callback=None):
    """
    A generator which takes *num_frames* using *camera*. *callback* is called
    after every taken frame.
    """
    if await camera.get_state() == 'recording':
        await camera.stop_recording()

    await camera['trigger_source'].stash()
    await camera.set_trigger_source(camera.trigger_sources.SOFTWARE)

    try:
        async with camera.recording():
            for i in range(num_frames):
                await camera.trigger()
                yield await camera.grab()
                if callback:
                    await callback()
    finally:
        await camera['trigger_source'].restore()


def tomo_angular_step(frame_width):
    """
    Get the angular step required for tomography so that every pixel of the frame
    rotates no more than one pixel per rotation step. *frame_width* is frame size in
    the direction perpendicular to the axis of rotation.
    """
    return np.arctan(2 / frame_width.magnitude) * q.rad


def tomo_projections_number(frame_width):
    """
    Get the minimum number of projections required by a tomographic scan in
    order to provide enough data points for every distance from the axis of
    rotation. The minimum angular step is
    considered to be needed smaller than one pixel in the direction
    perpendicular to the axis of rotation. The number of pixels in this
    direction is given by *frame_width*.
    """
    return int(np.ceil(np.pi / tomo_angular_step(frame_width)))


def tomo_max_speed(frame_width, frame_rate):
    """
    Get the maximum rotation speed which introduces motion blur less than one
    pixel. *frame_width* is the width of the frame in the direction
    perpendicular to the rotation and *frame_rate* defines the time required
    for recording one frame.

    _Note:_ frame rate is required instead of exposure time because the
    exposure time is usually shorter due to the camera chip readout time.
    We need to make sure that by the next exposure the sample hasn't moved
    more than one pixel from the previous frame, thus we need to take into
    account the whole frame taking procedure (exposure + readout).
    """
    return tomo_angular_step(frame_width) * frame_rate


class SynchrotronMixin:
    """
    Mixin to implement the required function start_sample_exposure and stop_sample_exposure in the imaging experiments
    for the synchrotron by opening and closing a shutter.
    """
    def __init__(self, shutter):
        self._shutter = shutter

    async def start_sample_exposure(self):
        if await self._shutter.get_state() != "open":
            await self._shutter.open()

    async def stop_sample_exposure(self):
        if await self._shutter.get_state() != "closed":
            await self._shutter.close()


class XrayTubeMixin:
    """
    Mixin to implement the required function start_sample_exposure and stop_sample_exposure in the imaging experiments
    by switching the X-ray tube on and off.
    """
    def __init__(self, xray_tube):
        self._xray_tube = xray_tube

    async def start_sample_exposure(self):
        if await self._xray_tube.get_state() != "on":
            await self._xray_tube.start()

    async def stop_sample_exposure(self):
        if await self._xray_tube.get_state() == "on":
            await self._xray_tube.stop()


class Radiography(Experiment):
    """
    Radiography experiment

    This records dark images (without beam) and flat images (with beam and without the sample) as well as the
    projections with the sample in the beam.
    """
    num_flats = Parameter()
    num_darks = Parameter()
    num_projections = Parameter()
    radio_position = Quantity(q.mm)
    flat_position = Quantity(q.mm)

    def __init__(self, walker, flat_motor, radio_position, flat_position, camera,
                 num_flats=200, num_darks=200, num_projections=3000, separate_scans=True):
        self._num_flats = num_flats
        self._num_darks = num_darks
        self._num_projections = num_projections
        self._radio_position = radio_position
        self._flat_position = flat_position
        self._finished = None
        self._flat_motor = flat_motor
        self._camera = camera
        darks_acq = Acquisition("darks", self._take_darks)
        flats_acq = Acquisition("flats", self._take_flats)
        radios_acq = Acquisition("radios", self._take_radios)
        super(Radiography, self).__init__([darks_acq, flats_acq, radios_acq], walker, separate_scans=separate_scans)

    async def _get_num_flats(self):
        return self._num_flats

    async def _get_num_darks(self):
        return self._num_darks

    async def _get_num_projections(self):
        return self._num_projections

    async def _get_radio_position(self):
        return self._radio_position

    async def _get_flat_position(self):
        return self._flat_position

    async def _set_num_flats(self, n):
        self._num_flats = int(n)

    async def _set_num_darks(self, n):
        self._num_darks = int(n)

    async def _set_num_projections(self, n):
        self._num_projections = int(n)

    async def _set_flat_position(self, position):
        self._flat_position = position

    async def _set_radio_position(self, position):
        self._radio_position = position

    async def _prepare_flats(self):
        await self._flat_motor.set_position(await self.get_flat_position())
        await self.start_sample_exposure()

    async def _prepare_darks(self):
        await self.stop_sample_exposure()

    async def _prepare_radios(self):
        await self._flat_motor.set_position(await self.get_radio_position())
        await self.start_sample_exposure()

    async def _finish_radios(self):
        """
        Function that is called after all frames are acquired. It will be called only once.
        """
        if self._finished:
            return
        await self.stop_sample_exposure()
        self._finished = True

    @background
    async def run(self):
        self._finished = False
        await super().run()

    async def _take_radios(self):
        try:
            await self._prepare_radios()
            async for frame in self._produce_frames(self._num_projections):
                yield frame
        finally:
            await self._finish_radios()

    async def _take_flats(self):
        try:
            await self._prepare_flats()
            async for frame in self._produce_frames(self._num_flats):
                yield frame
        finally:
            await self._prepare_darks()

    async def _take_darks(self):
        try:
            await self._prepare_darks()
            async for frame in self._produce_frames(self._num_darks):
                yield frame
        finally:
            await self._prepare_darks()

    async def _produce_frames(self, number, after_acquisition=None):
        """
        Generator of frames

        :param number: Number of frames that are generated
        :param after_acquisition: function that is called after all frames are acquired (but maybe not yet downloaded).
        Could be None or a Future.
        :return:
        """
        await self._camera.set_trigger_source("AUTO")
        async with self._camera.recording():
            for i in range(int(number)):
                yield await self._camera.grab()
        if after_acquisition is not None:
            await after_acquisition()

    async def start_sample_exposure(self):
        raise NotImplementedError()

    async def stop_sample_exposure(self):
        raise NotImplementedError()


class SteppedTomography(Radiography):
    """
    Stepped Tomography

    Implementation of a step-and-shoot tomography. Prior to the tomography dark and flat images are recorded as in the
    Radiography experiment.
    """
    angular_range = Quantity(q.deg)
    start_angle = Quantity(q.deg)

    def __init__(self, walker, flat_motor, tomography_motor, radio_position, flat_position, camera,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg,
                 separate_scans=True):
        self._angular_range = angular_range
        self._start_angle = start_angle
        self._tomography_motor = tomography_motor
        super(SteppedTomography, self).__init__(walker=walker, flat_motor=flat_motor,
                                                radio_position=radio_position, flat_position=flat_position,
                                                camera=camera, num_flats=num_flats, num_darks=num_darks,
                                                num_projections=num_projections, separate_scans=separate_scans)

    async def _get_angular_range(self):
        return self._angular_range

    async def _get_start_angle(self):
        return self._start_angle

    async def _set_angular_range(self, angle):
        self._angular_range = angle

    async def _set_start_angle(self, angle):
        self._start_angle = angle

    async def _prepare_radios(self):
        await self._tomography_motor.set_position(await self.get_start_angle())
        await super(SteppedTomography, self)._prepare_radios()

    async def _finish_radios(self):
        if self._finished:
            return
        await self._prepare_darks()
        await self._tomography_motor.set_position(await self.get_start_angle())
        self._finished = True

    async def _take_radios(self):
        try:
            await self._prepare_radios()
            await self._camera.set_trigger_source("SOFTWARE")
            async with self._camera.recording():
                for i in range(await self.get_num_projections()):
                    await self._tomography_motor.set_position(
                        i * await self.get_angular_range() / await self.get_num_projections() +
                        await self.get_start_angle()
                    )
                    try:
                        await self._camera.trigger()
                    except:
                        pass
                    yield await self._camera.grab()
        finally:
            await self._finish_radios()

    async def start_sample_exposure(self):
        raise NotImplementedError()

    async def stop_sample_exposure(self):
        raise NotImplementedError()


class ContinuousTomography(SteppedTomography):
    """
    Continuous Tomography

    This implements a tomography with a continuous rotation of the sample. The camera must record frames with a constant
    rate.
    """
    velocity = Quantity(q.deg / q.s)

    def __init__(self, walker, flat_motor, tomography_motor, radio_position, flat_position, camera,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg,
                 separate_scans=True):
        super(ContinuousTomography, self).__init__(walker=walker, flat_motor=flat_motor,
                                                   tomography_motor=tomography_motor,
                                                   radio_position=radio_position, flat_position=flat_position,
                                                   camera=camera, num_flats=num_flats, num_darks=num_darks,
                                                   num_projections=num_projections, angular_range=angular_range,
                                                   start_angle=start_angle, separate_scans=separate_scans)

    async def _get_velocity(self):
        angular_range = await self.get_angular_range()
        num_projections = await self.get_num_projections()
        fps = await self._camera.get_frame_rate()

        return fps * angular_range / num_projections

    async def _prepare_radios(self):
        await super(ContinuousTomography, self)._prepare_radios()
        if 'motion_velocity' in self._tomography_motor:
            await self._tomography_motor['motion_velocity'].stash()

    async def _finish_radios(self):
        if self._finished:
            return
        await self._prepare_darks()
        await self._tomography_motor.stop()
        if 'motion_velocity' in self._tomography_motor:
            await self._tomography_motor['motion_velocity'].restore()
        await self._tomography_motor.set_position(await self.get_start_angle())
        self._finished = True

    async def _take_radios(self):
        try:
            await self._prepare_radios()
            await self._tomography_motor.set_velocity(await self.get_velocity())

            async def callback():
                await self._finish_radios()
                if hasattr(self, 'callback'):
                    await self.callback()

            async for frame in self._produce_frames(await self.get_num_projections(), callback):
                yield frame
        finally:
            await self._finish_radios()

    async def start_sample_exposure(self):
        raise NotImplementedError()

    async def stop_sample_exposure(self):
        raise NotImplementedError()


class SpiralTomography(ContinuousTomography):
    """
    Spiral Tomography

    This implements a helical acquisition scheme, where the sample is translated perpendicular to the beam while the
    sample is rotated and the projections are recorded.
    """
    vertical_velocity = Quantity(q.mm / q.s)
    start_position_vertical = Quantity(q.mm)
    vertical_shift_per_tomogram = Quantity(q.mm)
    sample_height = Quantity(q.mm)
    num_tomograms = Parameter()

    def __init__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position, flat_position,
                 camera, start_position_vertical, sample_height, vertical_shift_per_tomogram,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg,
                 separate_scans=True):
        self._start_position_vertical = start_position_vertical
        self._vertical_shift_per_tomogram = vertical_shift_per_tomogram
        self._sample_height = sample_height
        self._vertical_motor = vertical_motor
        super(SpiralTomography, self).__init__(walker=walker, flat_motor=flat_motor, tomography_motor=tomography_motor,
                                               radio_position=radio_position,
                                               flat_position=flat_position,
                                               camera=camera, num_flats=num_flats, num_darks=num_darks,
                                               num_projections=num_projections, angular_range=angular_range,
                                               start_angle=start_angle,
                                               separate_scans=separate_scans)

    async def _get_start_position_vertical(self):
        return self._start_position_vertical

    async def _get_vertical_velocity(self):
        shift = await self.get_vertical_shift_per_tomogram()
        fps = await self._camera.get_frame_rate()
        num = await self.get_num_projections()

        return shift * fps / num

    async def _get_num_tomograms(self):
        shift = await self.get_vertical_shift_per_tomogram()
        height = await self.get_sample_height()

        return abs(height / shift).to_base_units().magnitude + 1

    async def _get_vertical_shift_per_tomogram(self):
        return self._vertical_shift_per_tomogram

    async def _get_sample_height(self):
        return self._sample_height

    async def _set_start_position_vertical(self, position):
        self._start_position_vertical = position

    async def _set_sample_height(self, height):
        self._sample_height = height

    async def _set_vertical_shift_per_tomogram(self, shift):
        self._vertical_shift_per_tomogram = shift

    async def _prepare_radios(self):
        await asyncio.gather(
            super()._prepare_radios(),
            self._vertical_motor.set_position(await self.get_start_position_vertical())
        )

    async def _finish_radios(self):
        if self._finished:
            return
        await self._prepare_darks()
        await self._tomography_motor.stop()
        await self._vertical_motor.stop()
        await self._vertical_motor['motion_velocity'].restore()
        await asyncio.gather(
            self._tomography_motor.set_position(await self.get_start_angle()),
            self._vertical_motor.set_position(await self.get_start_position_vertical())
        )
        self._finished = True

    async def _take_radios(self):
        try:
            await self._prepare_radios()
            await self._vertical_motor['motion_velocity'].stash()
            await self._tomography_motor.set_velocity(await self.get_velocity())
            await self._vertical_motor.set_velocity(await self.get_vertical_velocity())

            async def callback():
                await self._finish_radios()
                if hasattr(self, 'callback'):
                    await self.callback()

            num_projections = await self.get_num_projections() * await self.get_num_tomograms()
            async for frame in self._produce_frames(num_projections, callback):
                yield frame
        finally:
            await self._finish_radios()

    async def start_sample_exposure(self):
        raise NotImplementedError()

    async def stop_sample_exposure(self):
        raise NotImplementedError()


class SynchrotronRadiography(SynchrotronMixin, Radiography):
    def __init__(self, walker, flat_motor, radio_position, flat_position, camera, shutter, num_flats=200, num_darks=200,
                 num_projections=3000, separate_scans=True):
        SynchrotronMixin.__init__(self, shutter)
        Radiography.__init__(self, walker=walker,
                             flat_motor=flat_motor,
                             radio_position=radio_position,
                             flat_position=flat_position,
                             camera=camera, num_flats=num_flats,
                             num_darks=num_darks,
                             num_projections=num_projections,
                             separate_scans=separate_scans)


class SynchrotronSteppedTomography(SynchrotronMixin, SteppedTomography):
    def __init__(self, walker, flat_motor, tomography_motor, radio_position, flat_position, camera, shutter,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg,
                 start_angle=0 * q.deg, separate_scans=True):
        SynchrotronMixin.__init__(self, shutter)
        SteppedTomography.__init__(self, walker=walker,
                                   flat_motor=flat_motor,
                                   tomography_motor=tomography_motor,
                                   radio_position=radio_position,
                                   flat_position=flat_position,
                                   camera=camera,
                                   num_flats=num_flats,
                                   num_darks=num_darks,
                                   num_projections=num_projections,
                                   angular_range=angular_range,
                                   start_angle=start_angle,
                                   separate_scans=separate_scans)


class SynchrotronContinuousTomography(SynchrotronMixin, ContinuousTomography):
    def __init__(self, walker, flat_motor, tomography_motor, radio_position, flat_position, camera, shutter,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg,
                 start_angle=0 * q.deg, separate_scans=True):
        SynchrotronMixin.__init__(self, shutter)
        ContinuousTomography.__init__(self, walker=walker,
                                      flat_motor=flat_motor,
                                      tomography_motor=tomography_motor,
                                      radio_position=radio_position,
                                      flat_position=flat_position,
                                      camera=camera,
                                      num_flats=num_flats,
                                      num_darks=num_darks,
                                      num_projections=num_projections,
                                      angular_range=angular_range,
                                      start_angle=start_angle,
                                      separate_scans=separate_scans)


class SynchrotronSpiralTomography(SynchrotronMixin, SpiralTomography):
    def __init__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position, flat_position,
                 camera, shutter, start_position_vertical, sample_height, vertical_shift_per_tomogram,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg,
                 separate_scans=True):
        SynchrotronMixin.__init__(self, shutter)
        SpiralTomography.__init__(self, walker=walker,
                                  flat_motor=flat_motor,
                                  tomography_motor=tomography_motor,
                                  vertical_motor=vertical_motor,
                                  radio_position=radio_position,
                                  flat_position=flat_position,
                                  camera=camera,
                                  start_position_vertical=start_position_vertical,
                                  sample_height=sample_height,
                                  vertical_shift_per_tomogram=vertical_shift_per_tomogram,
                                  num_flats=num_flats,
                                  num_darks=num_darks,
                                  num_projections=num_projections,
                                  angular_range=angular_range,
                                  start_angle=start_angle,
                                  separate_scans=separate_scans)


class XrayTubeRadiography(XrayTubeMixin, Radiography):
    def __init__(self, walker, flat_motor, radio_position, flat_position, camera, xray_tube, num_flats=200,
                 num_darks=200, num_projections=3000, separate_scans=True):
        XrayTubeMixin.__init__(self, xray_tube)
        Radiography.__init__(self, walker=walker,
                             flat_motor=flat_motor,
                             radio_position=radio_position,
                             flat_position=flat_position,
                             camera=camera, num_flats=num_flats,
                             num_darks=num_darks,
                             num_projections=num_projections,
                             separate_scans=separate_scans)


class XrayTubeSteppedTomography(XrayTubeMixin, SteppedTomography):
    def __init__(self, walker, flat_motor, tomography_motor, radio_position, flat_position, camera, xray_tube,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg,
                 start_angle=0 * q.deg, separate_scans=True):
        XrayTubeMixin.__init__(self, xray_tube)
        SteppedTomography.__init__(self, walker=walker,
                                   flat_motor=flat_motor,
                                   tomography_motor=tomography_motor,
                                   radio_position=radio_position,
                                   flat_position=flat_position,
                                   camera=camera,
                                   num_flats=num_flats,
                                   num_darks=num_darks,
                                   num_projections=num_projections,
                                   angular_range=angular_range,
                                   start_angle=start_angle,
                                   separate_scans=separate_scans)


class XrayTubeContinuousTomography(XrayTubeMixin, ContinuousTomography):
    def __init__(self, walker, flat_motor, tomography_motor, radio_position, flat_position, camera, xray_tube,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg,
                 start_angle=0 * q.deg, separate_scans=True):
        XrayTubeMixin.__init__(self, xray_tube)
        ContinuousTomography.__init__(self, walker=walker,
                                      flat_motor=flat_motor,
                                      tomography_motor=tomography_motor,
                                      radio_position=radio_position,
                                      flat_position=flat_position,
                                      camera=camera,
                                      num_flats=num_flats,
                                      num_darks=num_darks,
                                      num_projections=num_projections,
                                      angular_range=angular_range,
                                      start_angle=start_angle,
                                      separate_scans=separate_scans)


class XrayTubeSpiralTomography(XrayTubeMixin, SpiralTomography):
    def __init__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position, flat_position,
                 camera, xray_tube, start_position_vertical, sample_height, vertical_shift_per_tomogram,
                 num_flats=200, num_darks=200, num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg,
                 separate_scans=True):
        XrayTubeMixin.__init__(self, xray_tube)
        SpiralTomography.__init__(self, walker=walker,
                                  flat_motor=flat_motor,
                                  tomography_motor=tomography_motor,
                                  vertical_motor=vertical_motor,
                                  radio_position=radio_position,
                                  flat_position=flat_position,
                                  camera=camera,
                                  start_position_vertical=start_position_vertical,
                                  sample_height=sample_height,
                                  vertical_shift_per_tomogram=vertical_shift_per_tomogram,
                                  num_flats=num_flats,
                                  num_darks=num_darks,
                                  num_projections=num_projections,
                                  angular_range=angular_range,
                                  start_angle=start_angle,
                                  separate_scans=separate_scans)
