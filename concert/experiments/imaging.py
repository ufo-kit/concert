"""Imaging experiments.

For all classes the abstract functions *start_sample_exposure* and *stop_sample_exposure* need to be
 implemented."""
import asyncio
import numpy as np
from concert.quantities import q
from concert.experiments.base import Experiment, Acquisition
from concert.base import background, Parameter, Quantity, Parameterizable, \
    AccessorNotImplementedError
from concert.base import check

from concert.experiments.base import _runnable_state


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


class Radiography(Experiment):
    """
    Radiography experiment

    This records dark images (without beam) and flat images (with beam and without the sample) as
    well as the projections with the sample in the beam.
    """

    num_flats = Parameter(check=check(source=_runnable_state))
    """Number of images acquired for flatfield correction."""

    num_darks = Parameter(check=check(source=_runnable_state))
    """Number of images acquired for dark correction."""

    num_projections = Parameter(check=check(source=_runnable_state))
    """Number of projection images."""

    num_projections_total = Parameter()
    """Total number of projections. For most of the experiments this is the same as the number of
    projections."""

    async def __ainit__(self, walker, flat_motor, radio_position, flat_position, camera, num_flats,
                        num_darks, num_projections, separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.cameras.base.Camera
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        """
        self._num_flats = None
        self._num_darks = None
        self._num_projections = None
        self._radio_position = None
        self._flat_position = None
        self._finished = None
        self._flat_motor = flat_motor
        self._camera = camera
        flat_motor_unit = self._flat_motor['position'].unit

        darks_acq = await Acquisition("darks", self._take_darks)
        flats_acq = await Acquisition("flats", self._take_flats)
        radios_acq = await Acquisition("radios", self._take_radios)
        await super().__ainit__([darks_acq, flats_acq, radios_acq], walker,
                                separate_scans=separate_scans)
        self.install_parameters(
            {"flat_position": Quantity(flat_motor_unit, check=check(source=_runnable_state)),
             "radio_position": Quantity(flat_motor_unit,
                                        check=check(source=_runnable_state))})
        await self.set_radio_position(radio_position)
        await self.set_flat_position(flat_position)
        await self.set_num_flats(num_flats)
        await self.set_num_darks(num_darks)
        await self.set_num_projections(num_projections)

    async def prepare(self):
        if await self._camera.get_state() != "standby":
            await self._camera.stop_recording()

    async def _last_acquisition_running(self) -> bool:
        return await self.acquisitions[-1].get_state() == "running"

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

    async def _get_num_projections_total(self):
        return await self.get_num_projections()

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
        """
        Called before flat images are acquired.

        The *flat_motor* is moved to the *flat_position* and :py:meth:`.start_sample_exposure()`
        will be called.
        """
        await self._flat_motor.set_position(await self.get_flat_position())
        await self.start_sample_exposure()

    async def _finish_flats(self):
        """
        Called after all flat images are acquired.

        Does nothing in this class.
        """
        pass

    async def _prepare_darks(self):
        """
        Called before the dark images are acquired.

        Calls :py:meth:`.stop_sample_exposure()`.
        """
        await self.stop_sample_exposure()

    async def _finish_darks(self):
        """
        Called after all dark images are acquired.

        Does nothing in this class.
        """
        pass

    async def _prepare_radios(self):
        """
        Called before the projection images are acquired.

        The *flat_motor* is moved to the *radio_position* and :py:meth:`.start_sample_exposure()`
        will be called.
        """
        await self._flat_motor.set_position(await self.get_radio_position())
        await self.start_sample_exposure()

    async def _finish_radios(self):
        """
        Function that is called after all frames are acquired. It will be called only once.

        This calls :py:meth:`.stop_sample_exposure()`.
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
        """
        Generator for projection images.

        First :py:meth:`._prepare_radios()` is called. Afterwards :py:meth:`._produce_frames()`
        generates the frames.
        At the end :py:meth:`._finish_radios()` is called.
        """
        try:
            await self._prepare_radios()
            async for frame in self._produce_frames(await self.get_num_projections_total()):
                yield frame
        finally:
            await self._finish_radios()

    async def _take_flats(self):
        """
        Generator for taking flatfield images

        First :py:meth:`._prepare_flats()` is called. Afterwards :py:meth:`._produce_frames()`
        generates the frames.
        At the end :py:meth:`._finish_flats()` is called.
        """
        try:
            await self._prepare_flats()
            async for frame in self._produce_frames(self._num_flats):
                yield frame
        finally:
            await self._finish_flats()

    async def _take_darks(self):
        """
        Generator for taking dark images

        First :py:meth:`._prepare_darks()` is called. Afterwards :py:meth:`._produce_frames()`
        generates the frames.
        At the end :py:meth:`._finish_darks()` is called.
        """
        try:
            await self._prepare_darks()
            async for frame in self._produce_frames(self._num_darks):
                yield frame
        finally:
            await self._finish_darks()

    async def _produce_frames(self, number, **kwargs):
        """
        Generator of frames.
        Sets the camera to auto-trigger and then grabs *number* of frames.

        :param number: Number of frames that are generated
        :type number: int
        """
        await self._camera.set_trigger_source("AUTO")
        async with self._camera.recording():
            for i in range(int(number)):
                yield await self._camera.grab()

    async def start_sample_exposure(self):
        """
        This function must implement in a way that the sample is exposed by radiation, like opening
        a shutter or starting an X-ray tube.
        """
        raise NotImplementedError

    async def stop_sample_exposure(self):
        """
        This function must implement in a way that the sample is not exposed by radiation, like
        closing a shutter or switching off an X-ray tube.
        """
        raise NotImplementedError


class Tomography(Radiography):
    """
       Tomography

       Abstract implementation of a tomography experiment.
       """

    angular_range = Quantity(q.deg, check=check(source=_runnable_state))
    """Range for scanning the *tomography_motor*."""

    start_angle = Quantity(q.deg, check=check(source=_runnable_state))
    """Initial position of the *tomography_motor*."""

    async def __ainit__(self, walker, flat_motor, tomography_motor, radio_position,
                        flat_position, camera, num_flats=200, num_darks=200, num_projections=3000,
                        angular_range=180 * q.deg, start_angle=0 * q.deg, separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param tomography_motor: RotationMotor for tomography.
        :type tomography_motor: concert.devices.motors.base.RotationMotor
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.cameras.base.Camera
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        :param angular_range: Range for the scan of the *tomography_motor*.
        :type angular_range: q.deg
        :param start_angle: Start position of *tomography_motor* for the first projection.
        :type start_angle: q.deg
        """
        self._angular_range = None
        self._start_angle = None
        self._tomography_motor = tomography_motor
        await super().__ainit__(
            walker=walker, flat_motor=flat_motor,
            radio_position=radio_position, flat_position=flat_position,
            camera=camera, num_flats=num_flats, num_darks=num_darks,
            num_projections=num_projections,
            separate_scans=separate_scans
        )
        await self.set_angular_range(angular_range)
        await self.set_start_angle(start_angle)

    async def _get_angular_range(self):
        return self._angular_range

    async def _get_start_angle(self):
        return self._start_angle

    async def _set_angular_range(self, angle):
        self._angular_range = angle

    async def _set_start_angle(self, angle):
        self._start_angle = angle

    async def _prepare_radios(self):
        """
        Called before the projection images are acquired.

        Moves the *tomography_motor* to the *start_angle* and calls
        :py:meth:`.Radiography._prepare_radios()`.
        """
        await asyncio.gather(self._tomography_motor.set_position(await self.get_start_angle()),
                             super()._prepare_radios())

    async def _finish_radios(self):
        """
        Called after the projection images are acquired.

        Moves the *tomography_motor* to the *start_angle* and calls
        :py:meth:`.Radiography._finish_radios()`.
        """
        if self._finished:
            return
        await asyncio.gather(self._tomography_motor.set_position(await self.get_start_angle()),
                             super()._finish_radios())
        self._finished = True

    async def _take_radios(self):
        """
        Abstract function for the generation of the projection images.
        This function is implemented for the different acquisition schemes in the subclasses
        :py:class:`SteppedTomography` and :py:class:`ContinuousTomography`.
        """
        raise NotImplementedError


class SteppedTomography(Tomography):
    """
    Stepped tomography experiment
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, radio_position,
                        flat_position, camera, num_flats=200, num_darks=200,
                        num_projections=3000, angular_range=180 * q.deg,
                        start_angle=0 * q.deg, separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.camera.base.Camera
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        :param angular_range: Range for the scan of the *tomography_motor*.
        :type angular_range: q.deg
        :param start_angle: Start position of *tomography_motor* for the first projection.
        :type start_angle: q.deg
        """
        await super().__ainit__(
            walker=walker, flat_motor=flat_motor,
            tomography_motor=tomography_motor,
            radio_position=radio_position,
            flat_position=flat_position,
            camera=camera, num_flats=num_flats,
            num_darks=num_darks,
            num_projections=num_projections,
            angular_range=angular_range,
            start_angle=start_angle,
            separate_scans=separate_scans
        )

    async def _prepare_frame(self, frame_number: int):
        """
        Prepares the next frame for acquisition. This function is called before a projection is
        triggered.
        :param frame_number:
        :return:
        """
        await self._tomography_motor.set_position(
            frame_number * await self.get_angular_range() / await self.get_num_projections()
            + await self.get_start_angle()
        )

    async def _take_radios(self):
        """
        Generator for projection images.

        First :py:meth:`._prepare_radios()` is called.
        The camera is set to software trigger.
        Then the tomography_motor will be moved to the positions
        i * angular_range / num_projections + start_angle for i = [0, num_projections-1].
        At each position one frame is triggered and grabbed.
        """
        try:
            await self._prepare_radios()
            await self._camera.set_trigger_source("SOFTWARE")
            async with self._camera.recording():
                for i in range(await self.get_num_projections_total()):
                    await self._prepare_frame(i)
                    await self._camera.trigger()
                    yield await self._camera.grab()
        finally:
            await self._finish_radios()


class ContinuousTomography(Tomography):
    """
    Continuous Tomography

    This implements a tomography with a continuous rotation of the sample. The camera must record
    frames with a constant rate.
    """
    velocity = Quantity(q.deg / q.s)
    """Velocity of the *tomography_motor* in the continuous scan."""

    async def __ainit__(self, walker, flat_motor, tomography_motor, radio_position, flat_position,
                        camera, num_flats=200, num_darks=200, num_projections=3000,
                        angular_range=180 * q.deg, start_angle=0 * q.deg, separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.camera.base.Camera
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        :param angular_range: Range for the scan of the *tomography_motor*.
        :type angular_range: q.deg
        :param start_angle: Start position of *tomography_motor* for the first projection.
        :type start_angle: q.deg
        """
        await Tomography.__ainit__(
            self, walker=walker, flat_motor=flat_motor,
            tomography_motor=tomography_motor,
            radio_position=radio_position, flat_position=flat_position,
            camera=camera, num_flats=num_flats, num_darks=num_darks,
            num_projections=num_projections, angular_range=angular_range,
            start_angle=start_angle, separate_scans=separate_scans
        )

    async def _get_velocity(self):
        angular_range = await self.get_angular_range()
        num_projections = await self.get_num_projections()
        fps = await self._camera.get_frame_rate()

        return fps * angular_range / num_projections

    async def _prepare_radios(self):
        """
        Called before projection images are acquired.

        Calls :py:meth:`.Tomography._prepare_radios()` and stashed the *motion_velocity* property of
        the *tomography_motor*.
        """
        if 'motion_velocity' in self._tomography_motor:
            await self._tomography_motor['motion_velocity'].stash()
        await super()._prepare_radios()
        await self._tomography_motor.set_velocity(await self.get_velocity())

    async def _finish_radios(self):
        """
        Called after the projections are acquired.

        Stops the continuous motion of the *tomography_motor* and restores the *motion_velocity*.
        Calls :py:meth:`.Tomography._finish_radios()` .
        """
        if self._finished:
            return
        if await self._tomography_motor.get_state() == "moving":
            await self._tomography_motor.stop()
        else:
            self.log.error("tomography_motor not in moving state after radios finished.")
        if 'motion_velocity' in self._tomography_motor:
            await self._tomography_motor['motion_velocity'].restore()
        await super()._finish_radios()
        self._finished = True

    async def _take_radios(self):
        """
        Generator for the projection images.

        The :py:meth:`_prepare_radios()` is called. Afterwards the velocity property of the
        tomography_motor is set to correct velocity.
        Then :py:meth:`_produce_frames()` will generate the images.
        At the end :py:meth:`_finish_radios()` is called.
        """
        try:
            await self._prepare_radios()

            async for frame in self._produce_frames(await self.get_num_projections_total()):
                yield frame
        finally:
            await self._finish_radios()


class SpiralMixin(Parameterizable):
    """
    Mixin for spiral tomography.
    """
    start_position_vertical = Quantity(q.mm, check=check(source=_runnable_state))
    """Initial position of the vertical motor."""

    vertical_shift_per_tomogram = Quantity(q.mm, check=check(source=_runnable_state))
    """Vertical shift per tomogram."""

    sample_height = Quantity(q.mm, check=check(source=['standby', 'error', 'cancelled']))
    """Height of the sample. *vertical_motor* will be scanned from *start_position_vertical* to
    *sample_height* + *vertical_shift_per_tomogram* to sample the whole specimen.
    """

    num_tomograms = Parameter()
    """Number of tomograms, that are required to cover the whole specimen."""

    num_projections_total = Parameter()
    """Total number of projection. This is equal to
    *num_tomograms* * *num_projections_per_tomogram*."""

    async def __ainit__(self, vertical_motor, start_position_vertical, sample_height,
                        vertical_shift_per_tomogram):
        """
        :param vertical_motor: LinearMotor to translate the sample along the tomographic axis.
        :type vertical_motor: concert.devices.motors.base.LinearMotor
        :param start_position_vertical: Start position of *vertical_motor*.
        :type start_position_vertical: q.mm
        :param sample_height: Height of the sample.
        :type sample_height: q.mm
        :param vertical_shift_per_tomogram: Distance *vertical_motor* is translated during one
            *angular_range*.
        :type vertical_shift_per_tomogram: q.mm
        """
        self._start_position_vertical = None
        self._vertical_shift_per_tomogram = None
        self._sample_height = None
        self._vertical_motor = vertical_motor
        await Parameterizable.__ainit__(self)
        await self.set_start_position_vertical(start_position_vertical)
        await self.set_vertical_shift_per_tomogram(vertical_shift_per_tomogram)
        await self.set_sample_height(sample_height)

    async def _get_start_position_vertical(self):
        return self._start_position_vertical

    async def _get_num_tomograms(self):
        shift = await self.get_vertical_shift_per_tomogram()
        height = await self.get_sample_height()

        return abs(height / shift).to_base_units().magnitude + 1

    async def _get_vertical_shift_per_tomogram(self):
        return self._vertical_shift_per_tomogram

    async def _get_sample_height(self):
        return self._sample_height

    async def _get_num_projections_total(self):
        return int(await self.get_num_tomograms() * await self.get_num_projections())

    async def _set_start_position_vertical(self, position):
        self._start_position_vertical = position

    async def _set_sample_height(self, height):
        self._sample_height = height

    async def _set_vertical_shift_per_tomogram(self, shift):
        self._vertical_shift_per_tomogram = shift


class SteppedSpiralTomography(SpiralMixin, SteppedTomography):
    """
    Stepped spiral tomography
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position,
                        flat_position, camera, start_position_vertical, sample_height,
                        vertical_shift_per_tomogram, num_flats=200, num_darks=200,
                        num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg,
                        separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param tomography_motor: RotationMotor for tomography scan.
        :type tomography_motor: concert.devices.motors.base.RotationMotor
        :param vertical_motor: LinearMotor to translate the sample along the tomographic axis.
        :type vertical_motor: concert.devices.motors.base.LinearMotor
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.cameras.base.Camera
        :param start_position_vertical: Start position of *vertical_motor*.
        :type start_position_vertical: q.mm
        :param sample_height: Height of the sample.
        :type sample_height: q.mm
        :param vertical_shift_per_tomogram: Distance *vertical_motor* is translated during one
            *angular_range*.
        :type vertical_shift_per_tomogram: q.mm
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        :param angular_range: Range for the scan of the *tomography_motor*.
        :type angular_range: q.deg
        :param start_angle: Start position of *tomography_motor* for the first projection.
        :type start_angle: q.deg
        """
        await SteppedTomography.__ainit__(
            self, walker=walker, flat_motor=flat_motor,
            tomography_motor=tomography_motor,
            radio_position=radio_position,
            flat_position=flat_position,
            camera=camera, num_flats=num_flats, num_darks=num_darks,
            num_projections=num_projections, angular_range=angular_range,
            start_angle=start_angle,
            separate_scans=separate_scans
        )
        await SpiralMixin.__ainit__(
            self,
            vertical_motor=vertical_motor,
            vertical_shift_per_tomogram=vertical_shift_per_tomogram,
            sample_height=sample_height,
            start_position_vertical=start_position_vertical
        )

    async def _prepare_frame(self, frame_number: int):
        vertical_step = (await self.get_vertical_shift_per_tomogram()
                         / await self.get_num_projections())
        angular_step = await self.get_angular_range() / await self.get_num_projections()
        rot_position = frame_number * angular_step + await self.get_start_angle()
        vertical_position = frame_number * vertical_step + await self.get_start_position_vertical()
        await asyncio.gather(self._tomography_motor.set_position(rot_position),
                             self._vertical_motor.set_position(vertical_position))

    async def _prepare_radios(self):
        """
        Prepares the radios.

        First :py:meth:`SteppedTomography._prepare_radios()` is called. Then the vertical_motor
        is moved to *start_position_vertical*.
        """
        await SteppedTomography._prepare_radios(self)
        await self._vertical_motor.set_position(await self.get_start_position_vertical())

    async def _finish_radios(self):
        """
        Called after the projections are acquired.

        First :py:meth:`SteppedTomorgaphy._finish_radios()` is called.
        Then the tomorgraphy_motor is moved to the start angle and the vertical_motor is moved to
        the start_position_vertical.
        """
        if self._finished:
            return
        await SteppedTomography._finish_radios(self)
        await asyncio.gather(
            self._tomography_motor.set_position(await self.get_start_angle()),
            self._vertical_motor.set_position(await self.get_start_position_vertical())
        )
        self._finished = True


class ContinuousSpiralTomography(SpiralMixin, ContinuousTomography):
    """
    Spiral Tomography

    This implements a helical acquisition scheme, where the sample is translated perpendicular to
    the beam while the sample is rotated and the projections are recorded.
    """
    vertical_velocity = Quantity(q.mm / q.s)

    async def __ainit__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position,
                        flat_position, camera, start_position_vertical, sample_height,
                        vertical_shift_per_tomogram, num_flats=200, num_darks=200,
                        num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg,
                        separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param tomography_motor: ContinuousRotationMotor for tomography scan.
        :type tomography_motor: concert.devices.motors.base.ContinuousRotationMotor
        :param vertical_motor: ContinuousLinearMotor to translate the sample along the tomographic
            axis.
        :type vertical_motor: concert.devices.motors.base.ContinuousLinearMotor
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.cameras.base.Camera
        :param start_position_vertical: Start position of *vertical_motor*.
        :type start_position_vertical: q.mm
        :param sample_height: Height of the sample.
        :type sample_height: q.mm
        :param vertical_shift_per_tomogram: Distance *vertical_motor* is translated during one
            *angular_range*.
        :type vertical_shift_per_tomogram: q.mm
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        :param angular_range: Range for the scan of the *tomography_motor*.
        :type angular_range: q.deg
        :param start_angle: Start position of *tomography_motor* for the first projection.
        :type start_angle: q.deg
        """
        await ContinuousTomography.__ainit__(
            self, walker=walker, flat_motor=flat_motor,
            tomography_motor=tomography_motor,
            radio_position=radio_position,
            flat_position=flat_position,
            camera=camera, num_flats=num_flats, num_darks=num_darks,
            num_projections=num_projections, angular_range=angular_range,
            start_angle=start_angle,
            separate_scans=separate_scans
        )
        await SpiralMixin.__ainit__(
            self, vertical_motor=vertical_motor,
            vertical_shift_per_tomogram=vertical_shift_per_tomogram,
            sample_height=sample_height,
            start_position_vertical=start_position_vertical
        )

    async def _get_vertical_velocity(self):
        shift = await self.get_vertical_shift_per_tomogram()
        fps = await self._camera.get_frame_rate()
        num = await self.get_num_projections()

        return shift * fps / num

    async def _prepare_radios(self):
        """
        Called before the projection images are acquired.

        Stashes motion velocities of *tomography_motor* and *vertical_motor*.
        Moves *tomography_motor* to the start position and *vertical_motor* to the start position.
        Moves *flat_motor* to the *radio_position*.
        Starts sample exposure.
        Starts motion of *tomography_motor* and *vertical_motor*.

        """
        if 'motion_velocity' in self._vertical_motor:
            await self._vertical_motor['motion_velocity'].stash()
        if 'motion_velocity' in self._tomography_motor:
            await self._tomography_motor['motion_velocity'].stash()

        await asyncio.gather(
            self._tomography_motor.set_position(await self.get_start_angle()),
            self._vertical_motor.set_position(await self.get_start_position_vertical()),
            self._flat_motor.set_position(await self.get_radio_position())
        )
        await self.start_sample_exposure()

        await asyncio.gather(self._tomography_motor.set_velocity(await self.get_velocity()),
                             self._vertical_motor.set_velocity(await self.get_vertical_velocity()))

    async def _finish_radios(self):
        """
        Called after the projection images are required.

        First this stops sample exposure, stops the tomography_motor, stops the vertical_motor.
        Then the *motion_velocity* of tomography_motor and vertical_motor is restored.
        """
        if self._finished:
            return

        await self.stop_sample_exposure()

        if await self._tomography_motor.get_state() == 'moving':
            await self._tomography_motor.stop()
        else:
            self.log.error("Tomography_motor not in moving state after radios finished.")
        if await self._vertical_motor.get_state() == 'moving':
            await self._vertical_motor.stop()
        else:
            self.log.error("Vertical_motor not in moving state after radios finished.")

        if 'motion_velocity' in self._tomography_motor:
            await self._tomography_motor['motion_velocity'].restore()
        if 'motion_velocity' in self._vertical_motor:
            await self._vertical_motor['motion_velocity'].restore()

        await asyncio.gather(
            self._tomography_motor.set_position(await self.get_start_angle()),
            self._vertical_motor.set_position(await self.get_start_position_vertical())
        )
        self._finished = True


class GratingInterferometryMixin(Parameterizable):
    """
    Mixin for grating interferometry specific experiments.
    """

    grating_period = Quantity(q.um, check=check(source=_runnable_state))
    """Period of the grating to scan."""

    num_periods = Parameter(check=check(source=_runnable_state))
    """Number of gratings period so scan."""

    num_steps_per_period = Parameter(check=check(source=_runnable_state))
    """Number of stepping positions per grating period."""

    stepping_start_position = Quantity(q.um, check=check(source=_runnable_state))
    """Position of the first grating step of *stepping_motor*"""

    propagation_distance = Quantity(q.mm, check=check(source=_runnable_state))
    """
    Distance between the sample and the analyzer grating.

    This is only used by the processing addon to calculate the phase shift.
    """
    async def __ainit__(self, stepping_motor, grating_period, stepping_start_position,
                        num_periods=1, num_steps_per_period=16, propagation_distance=None):
        """
        :param stepping_motor: Motor for the stepping of one of the gratings
        :type stepping_motor: LinearMotor
        :param grating_period: Period of the stepped grating
        :type grating_period: q.um
        :param stepping_start_position: First stepping position
        :type stepping_start_position: q.um
        :param num_periods: Number of periods that are sampled by the stepping
        :type num_periods: int
        :param num_steps_per_period: Number of stepping positions per period
        :type num_steps_per_period: int
        :param propagation_distance: Distance between the sample and the analyzer grating. Only used
            by the processing addon to determine the phase shift in angles.
        :type propagation_distance: q.mm
        """

        self._grating_period = None
        self._num_periods = None
        self._num_steps_per_period = None
        self._stepping_start_position = None
        self._propagation_distance = None

        await Parameterizable.__ainit__(self)
        self._stepping_motor = stepping_motor
        await self.set_grating_period(grating_period)
        await self.set_num_periods(num_periods)
        await self.set_num_steps_per_period(num_steps_per_period)
        await self.set_stepping_start_position(stepping_start_position)
        await self.set_propagation_distance(propagation_distance)

    async def _get_propagation_distance(self):
        return self._propagation_distance

    async def _set_propagation_distance(self, distance):
        self._propagation_distance = distance

    async def _get_grating_period(self):
        return self._grating_period

    async def _set_grating_period(self, period):
        self._grating_period = period

    async def _get_num_periods(self):
        return self._num_periods

    async def _set_num_periods(self, periods):
        self._num_periods = int(periods)

    async def _get_num_steps_per_period(self):
        return self._num_steps_per_period

    async def _set_num_steps_per_period(self, periods):
        self._num_steps_per_period = int(periods)

    async def _get_stepping_start_position(self):
        return self._stepping_start_position

    async def _set_stepping_start_position(self, position):
        self._stepping_start_position = position


class GratingInterferometryStepping(GratingInterferometryMixin, Radiography):
    """
    Grating interferometry experiment.

    Data can be automatically processed with the corresponding addon
    (concert.experiments.addons.PhaseGratingSteppingFourierProcessing).
    """

    async def __ainit__(self, walker, camera, flat_motor, stepping_motor, flat_position,
                        radio_position, grating_period, num_darks, stepping_start_position,
                        num_periods=1, num_steps_per_period=16, propagation_distance=None,
                        separate_scans=False):
        """
        :param walker: Walker for the experiment
        :type walker: concert.storage.DirectoryWalker
        :param camera: Camera to acquire the images
        :type camera: concert.devices.cameras.base.Camera
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param stepping_motor:
        :type stepping_motor: concert.devices.motors.base.LinearMotor
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param grating_period: Periodicity of the stepped grating.
        :type grating_period: q.um
        :param num_darks: Number of dark images that are acquired.
        :type num_darks: int
        :param stepping_start_position: First stepping position.
        :type stepping_start_position: q.um
        :param num_periods: Number of grating periods that are sampled by the stepping.
        :type num_periods: int
        :param num_steps_per_period: Number stepping positions per grating period.
        :type num_steps_per_period: int
        :param propagation_distance: Distance between the sample and the analyzer grating. Only used
            by the processing addon to determine the phase shift in angles.
        :type propagation_distance: q.mm
        """

        await Radiography.__ainit__(
            self, walker=walker, camera=camera, flat_motor=flat_motor,
            radio_position=radio_position,
            flat_position=flat_position,
            num_flats=0,
            num_darks=num_darks,
            num_projections=1, separate_scans=separate_scans
        )
        await GratingInterferometryMixin.__ainit__(
            self, stepping_motor=stepping_motor,
            grating_period=grating_period,
            stepping_start_position=stepping_start_position,
            num_periods=num_periods,
            num_steps_per_period=num_steps_per_period,
            propagation_distance=propagation_distance
        )

        reference_stepping = await Acquisition("reference_stepping", self._take_reference_scan)
        self.remove(self.get_acquisition("flats"))  # No flats required due to ref. stepping
        self.add(reference_stepping)

        object_stepping = await Acquisition("object_stepping", self._take_object_scan)
        self.remove(self.get_acquisition("radios"))  # No radios required due to obj. stepping
        self.add(object_stepping)

    async def _get_num_flats(self):
        return 0

    async def _set_num_flats(self, n):
        if n != 0:
            raise AccessorNotImplementedError

    async def _get_num_projections(self):
        return 1

    async def _set_num_projections(self, n):
        if n != 1:
            raise AccessorNotImplementedError

    async def _take_reference_scan(self):
        """
        Starts the sample exposure, moves the flat_motor in the *reference_position* and runs
            self._take_scan().
        """
        await self.start_sample_exposure()
        await self._flat_motor.set_position(await self.get_flat_position())
        async for image in self._take_scan():
            yield image

    async def _take_object_scan(self):
        """
        Starts the sample exposure, moves the flat_motor in the *radio_position* and runs
            self._take_scan().
        """
        await self.start_sample_exposure()
        await self._flat_motor.set_position(await self.get_radio_position())
        async for image in self._take_scan():
            yield image

    async def _take_scan(self):
        """
        Scans the stepping motor and acquires a frame after each position is reached.

        As step *size grating_period* / *num_steps_per_period* is used.
        A total of *num_steps_per_period* * *num_periods* frames is acquired.
        """
        step_size = await self.get_grating_period() / await self.get_num_steps_per_period()
        if await self._camera.get_state() != "standby":
            await self._camera.stop_recording()
        await self._camera.set_trigger_source("SOFTWARE")
        async with self._camera.recording():
            for i in range(await self.get_num_periods() * await self.get_num_steps_per_period()):
                await self._stepping_motor.set_position(i * step_size
                                                        + await self.get_stepping_start_position())
                await self._camera.trigger()
                yield await self._camera.grab()

    async def finish(self):
        """
        This function calls stop_sample_exposure() at the end of the experiment run.
        """
        await self.stop_sample_exposure()

    async def start_sample_exposure(self):
        """
        This function must implement in a way that the sample is exposed by radiation, like opening
        a shutter or starting an X-ray tube.
        """
        raise NotImplementedError

    async def stop_sample_exposure(self):
        """
        This function must implement in a way that the sample is not exposed by radiation, like
        closing a shutter or switching off an X-ray tube.
        """
        raise NotImplementedError
