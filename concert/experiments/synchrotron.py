"""
Module for synchrotron based imaging experiments.
"""
from concert.quantities import q
from concert.experiments import imaging


# Mixins


class SynchrotronMixin:
    """
    Mixin to implement the required function start_sample_exposure and stop_sample_exposure in the
    imaging experiments for the synchrotron by opening and closing a shutter.
    """

    # The class implementing this must assign a real device here
    _shutter = None

    async def start_sample_exposure(self):
        """
        Starts the sample exposure.

        This calls *shutter.open()*.
        """
        if await self._shutter.get_state() != "open":
            await self._shutter.open()

    async def stop_sample_exposure(self):
        """
        Stops the sample exposure.

        This calls *shutter.close()*.
        """
        if await self._shutter.get_state() != "closed":
            await self._shutter.close()


# Logic


class RadiographyLogic(SynchrotronMixin, imaging.RadiographyLogic):
    """
    Synchrotron radiography logic class which needs to be combined with one of the local or remote
    mixins for DAQ.
    """
    async def __ainit__(self, walker, flat_motor, radio_position, flat_position, camera,
                        shutter, num_flats=200, num_darks=200, num_projections=3000,
                        separate_scans=True):
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
        :param shutter: Shutter
        :type shutter: concert.devices.shutters.base.Shutter
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        :param separate_scans: Descend into subdirectories based on current iteration.
        :type separate_scans: bool
        """
        self._shutter = shutter
        await super().__ainit__(
            walker,
            flat_motor,
            radio_position,
            flat_position,
            camera,
            num_flats,
            num_darks,
            num_projections,
            separate_scans=separate_scans
        )


class ContinuousTomographyLogic(SynchrotronMixin, imaging.ContinuousTomographyLogic):
    """
    Continuous tomography logic class which needs to be combined with one of the local or remote
    mixins for DAQ.
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, radio_position, flat_position,
                        camera, shutter, num_flats=200, num_darks=200, num_projections=3000,
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
        :param shutter: Shutter
        :type shutter: concert.devices.shutters.base.Shutter
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
        self._shutter = shutter
        await super().__ainit__(
            walker,
            flat_motor,
            tomography_motor,
            radio_position,
            flat_position,
            camera,
            num_flats=num_flats,
            num_darks=num_darks,
            num_projections=num_projections,
            angular_range=angular_range,
            start_angle=start_angle,
            separate_scans=separate_scans
        )


class SteppedTomographyLogic(SynchrotronMixin, imaging.SteppedTomographyLogic):
    """
    Stepped tomography logic class which needs to be combined with one of the local or remote
    mixins for DAQ.
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, radio_position, flat_position,
                        camera, shutter, num_flats=200, num_darks=200, num_projections=3000,
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
        :param shutter: Shutter
        :type shutter: concert.devices.shutters.base.Shutter
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
        self._shutter = shutter
        await super().__ainit__(
            walker,
            flat_motor,
            tomography_motor,
            radio_position,
            flat_position,
            camera,
            num_flats=num_flats,
            num_darks=num_darks,
            num_projections=num_projections,
            angular_range=angular_range,
            start_angle=start_angle,
            separate_scans=separate_scans
        )


class ContinuousSpiralTomographyLogic(SynchrotronMixin, imaging.ContinuousSpiralTomographyLogic):
    """
    Continuous spiral tomography logic class which needs to be combined with one of the local or
    remote mixins for DAQ.
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position,
                        flat_position, camera, shutter, start_position_vertical, sample_height,
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
        :param shutter: Stutter
        :type shutter: concert.devices.shutters.base.Shutter
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
        self._shutter = shutter
        await super().__ainit__(
            walker,
            flat_motor,
            tomography_motor,
            vertical_motor,
            radio_position,
            flat_position,
            camera,
            start_position_vertical,
            sample_height,
            vertical_shift_per_tomogram,
            num_flats=num_flats,
            num_darks=num_darks,
            num_projections=num_projections,
            angular_range=angular_range,
            start_angle=start_angle,
            separate_scans=separate_scans
        )


class SteppedSpiralTomographyLogic(SynchrotronMixin, imaging.SteppedSpiralTomographyLogic):
    """
    Stepped spiral tomography logic class which needs to be combined with one of the local or
    remote mixins for DAQ.
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position,
                        flat_position, camera, shutter, start_position_vertical, sample_height,
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
        :param shutter: Stutter
        :type shutter: concert.devices.shutters.base.Shutter
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
        self._shutter = shutter
        await super().__ainit__(
            walker,
            flat_motor,
            tomography_motor,
            vertical_motor,
            radio_position,
            flat_position,
            camera,
            start_position_vertical,
            sample_height,
            vertical_shift_per_tomogram,
            num_flats=num_flats,
            num_darks=num_darks,
            num_projections=num_projections,
            angular_range=angular_range,
            start_angle=start_angle,
            separate_scans=separate_scans
        )


# Ready-to-use local and remote experiment implementations


class LocalRadiography(imaging.LocalAutoDAQMixin, RadiographyLogic):
    """
    Radiography with local DAQ.
    """
    pass


class RemoteRadiography(imaging.RemoteAutoDAQMixin, RadiographyLogic):
    """
    Radiography with remote DAQ.
    """
    pass


class LocalSteppedTomography(imaging.LocalTriggerDAQMixin, SteppedTomographyLogic):
    """
    Stepped tomography with local DAQ.
    """
    pass


class RemoteSteppedTomography(imaging.RemoteTriggerDAQMixin, SteppedTomographyLogic):
    """
    Stepped tomography with remote DAQ.
    """
    pass


class LocalContinuousTomography(imaging.LocalAutoDAQMixin, ContinuousTomographyLogic):
    """
    Continuous tomography with local DAQ.
    """
    pass


class RemoteContinuousTomography(imaging.RemoteAutoDAQMixin, ContinuousTomographyLogic):
    """
    Continuous tomography with remote DAQ.
    """
    pass


class LocalSteppedSpiralTomography(imaging.LocalTriggerDAQMixin, SteppedSpiralTomographyLogic):
    """
    Stepped spiral tomography with local DAQ.
    """
    pass


class RemoteSteppedSpiralTomography(imaging.RemoteTriggerDAQMixin, SteppedSpiralTomographyLogic):
    """
    Stepped spiral tomography with remote DAQ.
    """
    pass


class LocalContinuousSpiralTomography(imaging.LocalAutoDAQMixin, ContinuousSpiralTomographyLogic):
    """
    Continuous spiral tomography with local DAQ.
    """
    pass


class RemoteContinuousSpiralTomography(imaging.RemoteAutoDAQMixin, ContinuousSpiralTomographyLogic):
    """
    Continuous spiral tomography with remote DAQ.
    """
    pass


class LocalGratingInterferometryStepping(
    SynchrotronMixin,
    imaging.LocalGratingInterferometryStepping
):
    """
    Local synchrotron grating interferometry experiment.

    Data can be automatically processed with the corresponding addon
    (concert.experiments.addons.local.PhaseGratingSteppingFourierProcessing).
    """
    async def __ainit__(self, walker, camera, shutter, flat_motor, stepping_motor, flat_position,
                        radio_position, grating_period, num_darks, stepping_start_position,
                        num_periods, num_steps_per_period, propagation_distance, separate_scans):
        """
        :param walker: Walker for the experiment
        :type walker: concert.storage.DirectoryWalker
        :param camera: Camera to acquire the images
        :type camera: concert.devices.cameras.base.Camera
        :param shutter: Shutter
        :type shutter: concert.devices.shutters.base.Shutter
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
        self._shutter = shutter
        await super().__ainit__(
            walker,
            camera,
            flat_motor,
            stepping_motor,
            flat_position,
            radio_position,
            grating_period,
            num_darks,
            stepping_start_position,
            num_periods=num_periods,
            num_steps_per_period=num_steps_per_period,
            propagation_distance=propagation_distance,
            separate_scans=separate_scans
        )
