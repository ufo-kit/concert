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
from concert.base import Parameter
from concert.devices.base import Device, LinearCalibration


LOG = logging.getLogger(__name__)


class Motor(Device):

    """Base class for everything that moves.

    A motor is used with a *calibration* that conforms to the
    :class:`~.Calibration` interface to convert between user and device units.
    If *calibration* is not given, a default :class:`~.LinearCalibration`
    mapping one step to one millimeter with zero offset is assumed.

    .. py:attribute:: position

        Motor position
    """

    state = State(default='standby')

    def __init__(self):
        super(Motor, self).__init__()

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

    def _get_position(self):
        return self._calibration.to_user(self._get_device_position())

    @transition(source='*', target='standby', immediate='moving')
    def _set_position(self, position):
        self._set_device_position(self._calibration.to_device(position))

    def _get_device_position(self):
        raise NotImplementedError

    def _home(self):
        raise NotImplementedError

    def in_hard_limit(self):
        return self._in_hard_limit()

    def _in_hard_limit(self):
        raise NotImplementedError

    position = Parameter(unit=q.count,
                         conversion=q.m / q.count,
                         in_hard_limit=in_hard_limit)


class ContinuousMotor(Motor):

    """A movable on which one can set velocity.

    This class is inherently capable of discrete movement.

    """

    def in_velocity_hard_limit(self):
        return self._in_velocity_hard_limit()

    def _in_velocity_hard_limit(self):
        raise NotImplementedError

    position = Parameter(unit=q.count,
                         conversion=q.deg / q.count)

    velocity = Parameter(unit=q.count / q.s,
                         conversion=q.deg / q.count,
                         in_hard_limit=in_velocity_hard_limit)
