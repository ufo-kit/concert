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
    @check(source='*', target='standby')
    def home(self):
        """
        home()

        Home motor.
        """
        self._home()

    def _home(self):
        raise AccessorNotImplementedError

    def _stop(self):
        raise AccessorNotImplementedError


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

    position = Quantity(q.m, help="Position",
                        check=check(source=['hard-limit', 'standby'],
                                    target=['hard-limit', 'standby']))


class ContinuousLinearMotor(LinearMotor):

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

    velocity = Quantity(q.m / q.s, help="Linear velocity",
                        check=check(source=['hard-limit', 'standby', 'moving'],
                                    target=['moving', 'standby']))


class RotationMotor(_PositionMixin):

    """
    One-dimensional rotational motor.

    .. attribute:: position

        Position of the motor in angular units.
    """

    state = State(default='standby')

    def _get_state(self):
        raise NotImplementedError

    position = Quantity(q.deg, help="Angular position",
                        check=check(source=['hard-limit', 'standby'],
                                    target=['hard-limit', 'standby']))

    def __init__(self):
        super(RotationMotor, self).__init__()


class ContinuousRotationMotor(RotationMotor):

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
