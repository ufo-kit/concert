"""
Each motor is associated with a :class:`~.Calibration` that maps arbitrary
real-world coordinates to devices coordinates. When a calibration is associated
with an motor, the position can be changed with :meth:`Motor.set_position` and
:meth:`Motor.move`::

    from concert.devices.motors.dummy import Motor

    motor = Motor()

    motor.position = 2 * q.mm
    motor.move(-0.5 * q.mm)

As long as an motor is moving, :meth:`Motor.stop` will stop the motion.
"""
import logging
from concert.quantities import q
from concert.async import async
from concert.base import Quantity, State, check, AccessorNotImplementedError
from concert.devices.base import Device
from concert.config import MOTOR_VELOCITY_SAMPLING_TIME as dT
from time import sleep


LOG = logging.getLogger(__name__)


class _PositionMixin(Device):

    """Provide positional, discrete behaviour interface."""

    def __init__(self):
        super(_PositionMixin, self).__init__()

    @async
    def move(self, delta):
        """
        move(delta)

        Move motor by *delta* user units."""
        self.position += delta

    @async
    @check(source=['hard-limit', 'moving'], target='standby')
    def stop(self):
        """
        stop()

        Stop the motion."""
        self._stop()

    @async
    @check(source='*', target=['standby', 'hard-limit'])
    def home(self):
        """
        home()

        Home motor.
        """
        self._home()

    @check(source='disabled', target='standby')
    def enable(self):
        self._enable()

    @check(source='standby', target='disabled')
    def disable(self):
        self._disable()

    def _enable(self):
        raise AccessorNotImplementedError

    def _disable(self):
        raise AccessorNotImplementedError

    def _home(self):
        raise AccessorNotImplementedError

    def _stop(self):
        raise AccessorNotImplementedError

    def _abort(self):
        """Abort by default stops. If the motor provides a real emergency stop, _abort should be
        overriden.
        """
        self._stop()

    def _cancel_position(self):
        """Cancel position setting."""
        self._abort()


class LinearMotor(_PositionMixin):

    """
    One-dimensional linear motor.

    .. attribute:: position

        Position of the motor in length units.
    """

    def __init__(self):
        super(LinearMotor, self).__init__()

    def _get_state(self):
        raise NotImplementedError

    state = State(default='standby')

    position = Quantity(q.mm, help="Position",
                        check=check(source=['hard-limit', 'standby'],
                                    target=['hard-limit', 'standby']))


class _VelocityMixin(object):

    """
    Provides velocity calculation for continuous motors
    from subsequent position measures.

    Sampling time for velocity measurement can be set in
    concert.config.MOTOR_VELOCITY_SAMPLING_TIME.
    """

    def _cancel_velocity(self):
        self._abort()

    def _get_velocity(self):
        pos0 = self.position
        sleep(dT.to(q.s).magnitude)
        pos1 = self.position
        return (pos1-pos0)/dT


class ContinuousLinearMotor(LinearMotor, _VelocityMixin):

    """
    One-dimensional linear motor with adjustable velocity.

    .. attribute:: velocity

        Current velocity in length per time unit.
    """

    def __init__(self):
        super(ContinuousLinearMotor, self).__init__()

    def _get_state(self):
        raise NotImplementedError

    state = State(default='standby')

    velocity = Quantity(q.mm / q.s, help="Linear velocity",
                        check=check(source=['hard-limit', 'standby', 'moving'],
                                    target=['moving', 'standby']))


class RotationMotor(_PositionMixin):

    """
    One-dimensional rotational motor.

    .. attribute:: position

        Position of the motor in angular units.
    """

    def _get_state(self):
        raise NotImplementedError

    state = State(default='standby')

    position = Quantity(q.deg, help="Angular position",
                        check=check(source=['hard-limit', 'standby'],
                                    target=['hard-limit', 'standby']))

    def __init__(self):
        super(RotationMotor, self).__init__()


class ContinuousRotationMotor(RotationMotor, _VelocityMixin):

    """
    One-dimensional rotational motor with adjustable velocity.

    .. attribute:: velocity

        Current velocity in angle per time unit.
    """

    def __init__(self):
        super(ContinuousRotationMotor, self).__init__()

    def _get_state(self):
        raise NotImplementedError

    state = State(default='standby')

    velocity = Quantity(q.deg / q.s, help="Angular velocity",
                        check=check(source=['hard-limit', 'standby', 'moving'],
                                    target=['moving', 'standby']))
