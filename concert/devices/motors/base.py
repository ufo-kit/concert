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
from concert.helpers import async
from concert.fsm import State, transition
from concert.base import Parameter, HardLimitError
from concert.devices.base import Device


LOG = logging.getLogger(__name__)


class PositionMixin(Device):

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

    def _set_position_checked(self, position):
        """Set position with after-check for hard limit."""
        self._set_position(position)
        if self.in_hard_limit():
            raise HardLimitError("hard limit reached")

    def _home(self):
        raise NotImplementedError

    def _stop(self):
        raise NotImplementedError

    def in_hard_limit(self):
        return self._in_hard_limit()

    def _in_hard_limit(self):
        raise NotImplementedError


class ContinuousMixin(Device):

    def __init__(self):
        super(ContinuousMixin, self).__init__()

    def in_velocity_hard_limit(self):
        return self._in_velocity_hard_limit()

    def _in_velocity_hard_limit(self):
        raise NotImplementedError


class LinearMotor(PositionMixin):

    position = Parameter(unit=q.m,
                         source='standby', target='standby', immediate='moving',
                         fset=PositionMixin._set_position_checked)

    def __init__(self):
        super(LinearMotor, self).__init__()


class ContinuousLinearMotor(LinearMotor, ContinuousMixin):

    velocity = Parameter(unit=q.m / q.s,
                         in_hard_limit=ContinuousMixin.in_velocity_hard_limit)

    def __init__(self):
        super(ContinuousLinearMotor, self).__init__()


class RotationMotor(PositionMixin):

    position = Parameter(unit=q.deg,
                         source='standby', target='standby', immediate='moving',
                         in_hard_limit=PositionMixin.in_hard_limit)

    def __init__(self):
        super(RotationMotor, self).__init__()


class ContinuousRotationMotor(RotationMotor, ContinuousMixin):

    def __init__(self):
        super(ContinuousRotationMotor, self).__init__()

    velocity = Parameter(unit=q.deg / q.s,
                         in_hard_limit=ContinuousMixin.in_velocity_hard_limit)
