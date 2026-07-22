"""A Dummy positioner"""
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.detectors.dummy import Detector
from concert.devices.positioners import base
from concert.devices.positioners.imaging import Positioner as ImagingBasePositioner


async def get_axes():
    """Get all the dummy axes."""
    x_lin = await LinearMotor()
    y_lin = await LinearMotor()
    z_lin = await LinearMotor()
    x_rot = await RotationMotor()
    y_rot = await RotationMotor()
    z_rot = await RotationMotor()

    x_lin_axis = await base.Axis(coordinate='x', motor=x_lin)
    y_lin_axis = await base.Axis(coordinate='y', motor=y_lin, direction=-1)
    z_lin_axis = await base.Axis(coordinate='z', motor=z_lin)
    x_rot_axis = await base.Axis(coordinate='x', motor=x_rot)
    y_rot_axis = await base.Axis(coordinate='y', motor=y_rot)
    z_rot_axis = await base.Axis(coordinate='z', motor=z_rot, direction=-1)

    return x_lin_axis, y_lin_axis, z_lin_axis, x_rot_axis, y_rot_axis, z_rot_axis


class Positioner(base.Positioner):

    """A dummy positioner."""

    async def __ainit__(self, *, position=None, **kwargs):
        await super().__ainit__(axes=await get_axes(), position=position, **kwargs)


class ImagingPositioner(ImagingBasePositioner):

    """A dummy imaging positioner."""

    async def __ainit__(self, *, detector=None, position=None, **kwargs):
        if detector is None:
            detector = await Detector()
        await super().__ainit__(axes=await get_axes(), detector=detector, position=position, **kwargs)
