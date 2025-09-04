"""Motor Dummy."""
import asyncio
import logging
from concert.base import HardLimitError, Quantity
from concert.coroutines.base import start
from concert.devices.base import Device
from concert.quantities import q
from concert.devices.motors import base
import numpy as np

LOG = logging.getLogger(__name__)
MOVEMENT_TIME_STEPS = 0.01 * q.s


class _PositionMixin(object):
    async def __ainit__(self):
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
        finally:
            self._moving = False
            self._stop_evt.set()

    async def _get_position(self):
        return self._position

    async def _home(self):
        LOG.debug('start homing')
        await asyncio.sleep(1)
        await self.set_position(0 * q.mm)
        LOG.debug('homes')

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

    async def __ainit__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        await base.LinearMotor.__ainit__(self)
        await _PositionMixin.__ainit__(self)
        self._motion_velocity = 200000 * q.mm / q.s

        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit


class ContinuousLinearMotor(LinearMotor, base.ContinuousLinearMotor):

    """A continuous linear motor dummy."""

    async def __ainit__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        await base.ContinuousLinearMotor.__ainit__(self)
        await LinearMotor.__ainit__(self)
        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit

    async def _set_velocity(self, vel):
        if vel.magnitude > 0:
            await self.set_motion_velocity(vel)
            if await self['position'].get_upper():
                start(self.set_position(await self['position'].get_upper()))
            else:
                start(self.set_position(np.inf * q.mm))
        if vel.magnitude < 0:
            await self.set_motion_velocity(vel)
            if await self['position'].get_lower():
                start(self.set_position(await self['position'].get_lower()))
            else:
                start(self.set_position(-np.inf * q.mm))
        if vel.magnitude == 0:
            start(self.stop())


class RotationMotor(_PositionMixin, base.RotationMotor):

    """A rotational step motor dummy."""

    motion_velocity = Quantity(q.deg / q.s)

    async def __ainit__(self, upper_hard_limit=None, lower_hard_limit=None):
        await base.RotationMotor.__ainit__(self)
        await _PositionMixin.__ainit__(self)
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

    async def __ainit__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        await base.ContinuousRotationMotor.__ainit__(self)
        await RotationMotor.__ainit__(self)
        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit

    async def _set_velocity(self, vel):
        if vel.magnitude > 0:
            await self.set_motion_velocity(vel)
            if await self['position'].get_upper():
                start(self.set_position(await self['position'].get_upper()))
            else:
                start(self.set_position(np.inf * q.deg))
        if vel.magnitude < 0:
            await self.set_motion_velocity(vel)
            if await self['position'].get_lower():
                start(self.set_position(await self['position'].get_lower()))
            else:
                start(self.set_position(-np.inf * q.deg))
        if vel.magnitude == 0:
            start(self.stop())


class TomographyStage(Device):

    """
    Dummy tomography stage. Sample is in position [0, 0, 0] and the stack of motors is below it.
    Top-to-bottom order of motors:
        - parallel_motor_above and orthogonal_motor_above (in the same height)
        - tomo_motor
        - roll_motor
        - lamino_motor
        - vertical_motor_below
        - orthogonal_motor_below
    """

    async def __ainit__(self, lamino_motor_z_offset=None, roll_motor_z_offset=None):
        await super().__ainit__()

        self.lamino_motor_z_offset = lamino_motor_z_offset
        self.roll_motor_z_offset = roll_motor_z_offset
        if self.lamino_motor_z_offset is None:
            self.lamino_motor_z_offset = [0, 0, -30] * q.cm
        if self.roll_motor_z_offset is None:
            self.roll_motor_z_offset = [0, 0, -20] * q.cm

        self.vertical_motor_below = await ContinuousLinearMotor()
        self.orthogonal_motor_below = await ContinuousLinearMotor()
        self.parallel_motor_above = await ContinuousLinearMotor()
        self.orthogonal_motor_above = await ContinuousLinearMotor()
        self.lamino_motor = await ContinuousRotationMotor()
        self.roll_motor = await ContinuousRotationMotor()
        self.tomo_motor = await ContinuousRotationMotor()
        self._motors = [
            self.parallel_motor_above,
            self.vertical_motor_below,
            self.orthogonal_motor_above,
            self.orthogonal_motor_below,
            self.lamino_motor,
            self.roll_motor,
            self.tomo_motor
        ]

    async def home(self):
        await asyncio.gather(*[motor.set_position(0 * motor["position"].unit) for motor in self._motors])
