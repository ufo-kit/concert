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
import asyncio
import logging
from concert.coroutines.base import background
from concert.quantities import q
from concert.base import Quantity, State, check, AccessorNotImplementedError
from concert.devices.base import Device
from concert.config import MOTOR_VELOCITY_SAMPLING_TIME as dT


LOG = logging.getLogger(__name__)


class _PositionMixin(Device):

    """Provide positional, discrete behaviour interface."""

    async def __ainit__(self):
        await super(_PositionMixin, self).__ainit__()

    @background
    async def move(self, delta):
        """
        move(delta)

        Move motor by *delta* user units."""
        await self.set_position(await self.get_position() + delta)

    @background
    @check(source=['hard-limit', 'moving'], target='standby')
    async def stop(self):
        """
        stop()

        Stop the motion."""
        await self._stop()

    @background
    @check(source='*', target=['standby', 'hard-limit'])
    async def home(self):
        """
        home()

        Home motor.
        """
        await self._home()

    @background
    @check(source='disabled', target='standby')
    async def enable(self):
        await self._enable()

    @background
    @check(source='standby', target='disabled')
    async def disable(self):
        await self._disable()

    async def _enable(self):
        raise AccessorNotImplementedError

    async def _disable(self):
        raise AccessorNotImplementedError

    async def _home(self):
        raise AccessorNotImplementedError

    async def _stop(self):
        raise AccessorNotImplementedError

    async def _emergency_stop(self):
        """Abort by default stops. If the motor provides a real emergency stop, _abort should be
        overriden.
        """
        if await self.get_state() == 'moving':
            await self.stop()


class LinearMotor(_PositionMixin):

    """
    One-dimensional linear motor.

    .. attribute:: position

        Position of the motor in length units.
    """

    async def __ainit__(self):
        await super(LinearMotor, self).__ainit__()

    async def _get_state(self):
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

    async def _get_velocity(self):
        pos0 = await self.get_position()
        await asyncio.sleep(dT.to(q.s).magnitude)
        pos1 = await self.get_position()

        return (pos1 - pos0) / dT


class ContinuousLinearMotor(LinearMotor, _VelocityMixin):

    """
    One-dimensional linear motor with adjustable velocity.

    .. attribute:: velocity

        Current velocity in length per time unit.
    """

    async def __ainit__(self):
        await super(ContinuousLinearMotor, self).__ainit__()

    async def _get_state(self):
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

    async def _get_state(self):
        raise NotImplementedError

    state = State(default='standby')

    position = Quantity(q.deg, help="Angular position",
                        check=check(source=['hard-limit', 'standby'],
                                    target=['hard-limit', 'standby']))

    async def __ainit__(self):
        await super(RotationMotor, self).__ainit__()


class ContinuousRotationMotor(RotationMotor, _VelocityMixin):

    """
    One-dimensional rotational motor with adjustable velocity.

    .. attribute:: velocity

        Current velocity in angle per time unit.
    """

    async def __ainit__(self):
        await super(ContinuousRotationMotor, self).__ainit__()

    async def _get_state(self):
        raise NotImplementedError

    state = State(default='standby')

    velocity = Quantity(q.deg / q.s, help="Angular velocity",
                        check=check(source=['hard-limit', 'standby', 'moving'],
                                    target=['moving', 'standby']))
