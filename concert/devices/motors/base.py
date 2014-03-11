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
from concert.base import Quantity, State, transition
from concert.devices.base import Device


LOG = logging.getLogger(__name__)


class PositionMixin(Device):

    """Provide positional, discrete behaviour interface."""

    state = State(default='standby')

    def __init__(self):
        super(PositionMixin, self).__init__()

    @async
    def move(self, delta):
        """
        move(delta)

        Move motor by *delta* user units."""
        self.position += delta

    @async
    @transition(source='moving', target='standby')
    def stop(self):
        """
        stop()

        Stop the motion."""
        self._stop()

    @async
    @transition(source='*', target='standby', immediate='moving')
    def home(self):
        """
        home()

        Home motor.
        """
        self._home()

    def _home(self):
        raise NotImplementedError

    def _stop(self):
        raise NotImplementedError


class ContinuousMixin(Device):

    """Provide continuous, non-discrete behaviour interface."""

    def __init__(self):
        super(ContinuousMixin, self).__init__()


class LinearMotor(PositionMixin):

    """
    One-dimensional linear motor.

    .. attribute:: position

        Position of the motor in length units.
    """

    def __init__(self):
        super(LinearMotor, self).__init__()

    def check_state(self):
        raise NotImplementedError

    state = State(default='standby')

    position = Quantity(unit=q.m,
                        transition=transition(source='standby',
                                              target=['standby'],
                                              immediate='moving',
                                              check=check_state))


class ContinuousLinearMotor(LinearMotor, ContinuousMixin):

    """
    One-dimensional linear motor with adjustable velocity.

    .. attribute:: velocity

        Current velocity in length per time unit.
    """

    def __init__(self):
        super(ContinuousLinearMotor, self).__init__()

    def check_state(self):
        raise NotImplementedError

    state = State(default='standby')

    velocity = Quantity(unit=q.m / q.s,
                        transition=transition(source=['standby', 'moving'],
                                              target=['moving', 'standby'],
                        check=check_state))


class RotationMotor(PositionMixin):

    """
    One-dimensional rotational motor.

    .. attribute:: position

        Position of the motor in angular units.
    """

    state = State(default='standby')

    position = Quantity(unit=q.deg,
                        transition=transition(source='standby',
                                              target='standby',
                                              immediate='moving'))

    def __init__(self):
        super(RotationMotor, self).__init__()


class ContinuousRotationMotor(RotationMotor, ContinuousMixin):

    """
    One-dimensional rotational motor with adjustable velocity.

    .. attribute:: velocity

        Current velocity in angle per time unit.
    """

    def __init__(self):
        super(ContinuousRotationMotor, self).__init__()

    def check_state(self):
        raise NotImplementedError

    state = State(default='standby')

    velocity = Quantity(unit=q.deg / q.s,
                        transition=transition(source=['standby', 'moving'],
                                              target=['moving', 'standby'],
                        check=check_state))
