"""A Dummy positioner"""
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.detectors.dummy import Detector
from concert.devices.positioners import base
from concert.devices.positioners.imaging import Positioner as ImagingBasePositioner


def get_axes():
    """Get all the dummy axes."""
    x_lin = LinearMotor()
    y_lin = LinearMotor()
    z_lin = LinearMotor()
    x_rot = RotationMotor()
    y_rot = RotationMotor()
    z_rot = RotationMotor()

    x_lin_axis = base.Axis('x', x_lin)
    y_lin_axis = base.Axis('y', y_lin, direction=-1)
    z_lin_axis = base.Axis('z', z_lin)
    x_rot_axis = base.Axis('x', x_rot)
    y_rot_axis = base.Axis('y', y_rot)
    z_rot_axis = base.Axis('z', z_rot, direction=-1)

    return (x_lin_axis, y_lin_axis, z_lin_axis, x_rot_axis, y_rot_axis, z_rot_axis)


class Positioner(base.Positioner):

    """A dummy positioner."""

    def __init__(self, position=None):
        super(Positioner, self).__init__(get_axes(), position=position)


class ImagingPositioner(ImagingBasePositioner):

    """A dummy imaging positioner."""

    def __init__(self, detector=None, position=None):
        if detector is None:
            detector = Detector()
        super(ImagingPositioner, self).__init__(get_axes(), detector, position=position)
