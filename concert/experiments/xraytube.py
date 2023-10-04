"""
Module for X-ray tube based imaging experiments.
"""
from concert.quantities import q
from concert.experiments.imaging import Radiography as BaseRadiography, \
    SteppedTomography as BaseSteppedTomo, \
    ContinuousTomography as BaseContTomo, \
    ContinuousSpiralTomography as BaseContSpiralTomo, \
    SteppedSpiralTomography as BaseSteppedSpiralTomo
from concert.experiments.imaging import GratingInterferometryStepping as BasePhaseStepping


class XrayTubeMixin:
    """
    Mixin to implement the required function start_sample_exposure and stop_sample_exposure in the
    imaging experiments by switching the X-ray tube on and off.
    """

    def __init__(self, xray_tube):
        """
        :param xray_tube: X-ray tube
        :type xray_tube: concert.devices.xraytubes.base.XRayTube
        """
        self._xray_tube = xray_tube

    async def start_sample_exposure(self):
        """
        Starts the sample exposure.

        The on() function of the xray_tube is called.
        """
        if await self._xray_tube.get_state() != "on":
            await self._xray_tube.on()

    async def stop_sample_exposure(self):
        """
        Stops the sample exposure.

        The off() function of the xray_tube is called.
        """
        if await self._xray_tube.get_state() == "on":
            await self._xray_tube.off()


class Radiography(XrayTubeMixin, BaseRadiography):
    """
    Radiography experiment
    """
    async def __ainit__(self, walker, flat_motor, radio_position, flat_position, camera, xray_tube,
                        num_flats=200, num_darks=200, num_projections=3000, separate_scans=True):
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
        :param xray_tube: X-ray tube
        :type xray_tube: concert.devices.xraytubes.base.XRayTube
        :param num_flats: Number of images for flatfield correction.
        :type num_flats: int
        :param num_darks: Number of images for dark correction.
        :type num_darks: int
        :param num_projections: Number of projections.
        :type num_projections: int
        """
        XrayTubeMixin.__init__(self, xray_tube)
        await BaseRadiography.__ainit__(
            self, walker=walker,
            flat_motor=flat_motor,
            radio_position=radio_position,
            flat_position=flat_position,
            camera=camera, num_flats=num_flats,
            num_darks=num_darks,
            num_projections=num_projections,
            separate_scans=separate_scans
        )


class SteppedTomography(XrayTubeMixin, BaseSteppedTomo):
    """
    Stepped tomography experiment.
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, radio_position, flat_position,
                        camera, xray_tube, num_flats=200, num_darks=200, num_projections=3000,
                        angular_range=360 * q.deg, start_angle=0 * q.deg, separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param tomography_motor: RotationMotor for tomography scan.
        :type tomography_motor: concert.devices.motors.base.RotationMotor
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.camera.base.Camera
        :param xray_tube: X-ray tube
        :type xray_tube: concert.devices.xraytubes.base.XRayTube
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
        XrayTubeMixin.__init__(self, xray_tube)
        await BaseSteppedTomo.__ainit__(
            self, walker=walker,
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
            separate_scans=separate_scans
        )


class ContinuousTomography(XrayTubeMixin, BaseContTomo):
    """
    Continuous tomography experiment
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, radio_position, flat_position,
                        camera, xray_tube, num_flats=200, num_darks=200, num_projections=3000,
                        angular_range=180 * q.deg, start_angle=0 * q.deg, separate_scans=True):
        """
        :param walker: Walker for storing experiment data.
        :type walker: concert.storage.Walker
        :param flat_motor: Motor for moving sample in and out of the beam. Must feature a
            'position' property.
        :param tomography_motor: ContinuousRotationMotor for tomography scan.
        :type tomography_motor: concert.devices.motors.base.ContinuousRotationMotor
        :param radio_position: Position of *flat_motor* that the sample is positioned in the beam.
            Unit must be the same as flat_motor['position'].
        :param flat_position: Position of *flat_motor* that the sample is positioned out of the
            beam. Unit must be the same as flat_motor['position'].
        :param camera: Camera to acquire the images.
        :type camera: concert.devices.camera.base.Camera
        :param xray_tube: X-ray tube
        :type xray_tube: concert.devices.xraytubes.base.XRayTube
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
        XrayTubeMixin.__init__(self, xray_tube)
        await BaseContTomo.__ainit__(
            self, walker=walker,
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
            separate_scans=separate_scans
        )


class ContinuousSpiralTomography(XrayTubeMixin, BaseContSpiralTomo):
    """
    Continuous spiral tomography experiment
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, vertical_motor, radio_position,
                        flat_position, camera, xray_tube, start_position_vertical, sample_height,
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
        :param xray_tube: X-ray tube
        :type xray_tube: concert.devices.xraytubes.base.XRayTube
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
        XrayTubeMixin.__init__(self, xray_tube)
        await BaseContSpiralTomo.__ainit__(
            self, walker=walker,
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
            separate_scans=separate_scans
        )


class SteppedSpiralTomography(XrayTubeMixin, BaseSteppedSpiralTomo):
    """
    Stepped spiral tomography experiment
    """
    async def __ainit__(self, walker, flat_motor, tomography_motor, vertical_motor,
                        radio_position, flat_position, camera, xray_tube,
                        start_position_vertical, sample_height, vertical_shift_per_tomogram,
                        num_flats=200, num_darks=200, num_projections=3000,
                        angular_range=180 * q.deg, start_angle=0 * q.deg, separate_scans=True):
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
        :param xray_tube: X-ray tube
        :type xray_tube: concert.devices.xraytubes.base.XRayTube
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
        XrayTubeMixin.__init__(self, xray_tube)
        await BaseSteppedSpiralTomo.__ainit__(
            self, walker=walker,
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
            separate_scans=separate_scans
        )


class GratingInterferometryStepping(XrayTubeMixin, BasePhaseStepping):
    async def __ainit__(self, walker, camera, xray_tube, flat_motor, stepping_motor,
                        flat_position, radio_position, grating_period, num_darks,
                        stepping_start_position, num_periods, num_steps_per_period,
                        propagation_distance, separate_scans):
        """
        :param walker: Walker for the experiment
        :type walker: concert.storage.DirectoryWalker
        :param camera: Camera to acquire the images
        :type camera: concert.devices.cameras.base.Camera
        :param xray_tube: Xray tube
        :type xray_tube: concert.devices.xraytubes.base.XRayTube
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
        XrayTubeMixin.__init__(self, xray_tube)
        await BasePhaseStepping.__ainit__(
            self, walker=walker,
            camera=camera,
            flat_motor=flat_motor,
            stepping_motor=stepping_motor,
            flat_position=flat_position,
            radio_position=radio_position,
            grating_period=grating_period,
            num_darks=num_darks,
            stepping_start_position=stepping_start_position,
            num_periods=num_periods,
            num_steps_per_period=num_steps_per_period,
            propagation_distance=propagation_distance,
            separate_scans=separate_scans
        )
