"""Motor Dummy."""
import asyncio
from concert.base import HardLimitError, Quantity
from concert.coroutines.base import start
from concert.quantities import q
from concert.devices.motors import base
import numpy as np

MOVEMENT_TIME_STEPS = 0.01 * q.s


class _PositionMixin(object):
    def __init__(self):
        self._position = 0 * q.mm
        self._moving = False
        self._stop_evt = asyncio.Event()
        self._lower_hard_limit = -np.inf * q.mm
        self._upper_hard_limit = np.inf * q.mm

    async def _set_position(self, position):
        self._stop_evt.clear()
        try:
            direction = 0
            motion_velocity = await self.get_motion_velocity()
            if self._position < position:
                direction = 1
            elif self._position > position:
                direction = -1

            if direction:
                self._moving = True
                while (direction * self._position) < (direction * position) and self._moving:
                    self._position += direction * motion_velocity * MOVEMENT_TIME_STEPS
                    await asyncio.sleep(MOVEMENT_TIME_STEPS.to(q.s).magnitude)
                    if self._position < self._lower_hard_limit:
                        self._position = self._lower_hard_limit
                        raise HardLimitError('hard-limit')
                    if self._position > self._upper_hard_limit:
                        self._position = self._upper_hard_limit
                        raise HardLimitError('hard-limit')
                if self._moving:
                    self._position = position
        except asyncio.CancelledError:
            # fall-through to finally
            pass
        finally:
            self._moving = False
            self._stop_evt.set()

    async def _get_position(self):
        return self._position

    async def _home(self):
        print('start homing')
        await asyncio.sleep(1)
        await self.set_position(0 * q.mm)
        print('homes')

    async def _stop(self):
        self._moving = False
        await self._stop_evt.wait()

    async def _set_motion_velocity(self, vel):
        self._motion_velocity = vel

    async def _get_motion_velocity(self):
        return self._motion_velocity

    async def _get_state(self):
        if self._moving:
            return 'moving'
        if self._position > self._lower_hard_limit and self._position < self._upper_hard_limit:
            return 'standby'
        return 'hard-limit'


class LinearMotor(_PositionMixin, base.LinearMotor):

    """A linear step motor dummy."""

    motion_velocity = Quantity(q.mm / q.s)

    def __init__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        base.LinearMotor.__init__(self)
        _PositionMixin.__init__(self)
        self._motion_velocity = 200000 * q.mm / q.s

        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit


class ContinuousLinearMotor(LinearMotor, base.ContinuousLinearMotor):

    """A continuous linear motor dummy."""

    def __init__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        base.ContinuousLinearMotor.__init__(self)
        LinearMotor.__init__(self)
        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit

    async def _set_velocity(self, vel):
        if vel.magnitude > 0:
            await self.set_motion_velocity(vel)
            if self['position'].upper:
                start(self.set_position(self['position'].upper))
            else:
                start(self.set_position(np.inf * q.mm))
        if vel.magnitude < 0:
            await self.set_motion_velocity(vel)
            if self['position'].lower:
                start(self.set_position(self['position'].lower))
            else:
                start(self.set_position(-np.inf * q.mm))
        if vel.magnitude == 0:
            start(self.stop())


class RotationMotor(_PositionMixin, base.RotationMotor):

    """A rotational step motor dummy."""

    motion_velocity = Quantity(q.deg / q.s)

    def __init__(self, upper_hard_limit=None, lower_hard_limit=None):
        base.RotationMotor.__init__(self)
        _PositionMixin.__init__(self)
        self._position = 0 * q.deg
        self._lower_hard_limit = -np.inf * q.deg
        self._upper_hard_limit = np.inf * q.deg
        self._motion_velocity = 50000 * q.deg / q.s
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit


class ContinuousRotationMotor(RotationMotor,
                              base.ContinuousRotationMotor):

    """A continuous rotational step motor dummy."""

    def __init__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        base.ContinuousRotationMotor.__init__(self)
        RotationMotor.__init__(self)
        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit

    async def _set_velocity(self, vel):
        if vel.magnitude > 0:
            await self.set_motion_velocity(vel)
            if self['position'].upper:
                start(self.set_position(self['position'].upper))
            else:
                start(self.set_position(np.inf * q.deg))
        if vel.magnitude < 0:
            await self.set_motion_velocity(vel)
            if self['position'].lower:
                start(self.set_position(self['position'].lower))
            else:
                start(self.set_position(-np.inf * q.deg))
        if vel.magnitude == 0:
            start(self.stop())
