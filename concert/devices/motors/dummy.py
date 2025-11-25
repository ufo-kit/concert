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

    """A motor which simulates the travel to a destination. If *backlash* is given, the motor
    includes a backlash when changing direction. In the beginning and after homing, the direction is
    positive, position is 0, and the actual position (considering backlash) is also 0.
    """

    async def __ainit__(
        self, position=None, upper_hard_limit=None, lower_hard_limit=None, backlash=None
    ):
        unit = self["position"].unit
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        else:
            self._upper_hard_limit = np.inf * unit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit
        else:
            self._lower_hard_limit = -np.inf * unit

        # Position-related
        if position:
            if position > self._upper_hard_limit or position < self._lower_hard_limit:
                raise ValueError("Position not within hard limits")
            self._position = position
        else:
            self._position = 0 * unit
        self._backlash = backlash
        self._current_backlash = None
        self._reset_backlash()

        self._moving = False
        self._stop_evt = asyncio.Event()

    async def _set_position(self, position):
        self._stop_evt.clear()
        try:
            direction = 0
            motion_velocity = await self.get_motion_velocity()
            if self._position < position:
                direction = 1
            elif self._position > position:
                direction = -1
            last_position = self._position
            diff = position - last_position

            if direction:
                self._moving = True
                while (direction * self._position) < (direction * position) and self._moving:
                    self._position += direction * motion_velocity * MOVEMENT_TIME_STEPS
                    if self._backlash is None:
                        # No backlash, no discrepancy between motor position and actual mechanism position
                        self._actual_position = self._position
                    await asyncio.sleep(MOVEMENT_TIME_STEPS.to(q.s).magnitude)
                    if self._position < self._lower_hard_limit:
                        # Do not use self._position which may be off at this point
                        # difference for backlash is taken from the limit
                        diff = self._lower_hard_limit - last_position
                        self._position = self._lower_hard_limit
                        raise HardLimitError('hard-limit')
                    if self._position > self._upper_hard_limit:
                        # Do not use self._position which may be off at this point
                        # difference for backlash is taken from the limit
                        diff = self._upper_hard_limit - last_position
                        self._position = self._upper_hard_limit
                        raise HardLimitError('hard-limit')
                if self._moving:
                    self._position = position
        finally:
            self._moving = False
            self._stop_evt.set()
            if self._backlash is None:
                # No backlash, no discrepancy between motor position and actual mechanism position
                self._actual_position = self._position
            else:
                if direction != self._last_direction:
                    # Direction change -> current backlash is the opposite of what it was before,
                    # this allows us to move back and forth by a tiny amount below backlash and
                    # cause no actual mechanism motion because of back and forth operation within
                    # the backlash of the mechanism
                    self._current_backlash = self._backlash - self._current_backlash
                # This is the actual backlash decrease amount
                backlash_amount = min(abs(diff), self._current_backlash)
                # Actual mechanism position is given by the difference of the motor position
                # plus/minus the backlash amount at this particular moment
                self._actual_position += diff - direction * backlash_amount
                # The next motion (if in the same direction) will have less backlash amount to
                # account for (if the movement here is smaller than the backlash value)
                self._current_backlash -= backlash_amount

            self._last_direction = direction

    def _reset_backlash(self):
        """By default the motor is moving positive, actual position is the same as position and no
        more backlash occurs if we move positive.
        """
        self._last_direction = 1
        self._actual_position = self._position
        if self._backlash is not None:
            self._current_backlash = 0 * self._backlash.units

    async def _get_position(self):
        return self._position

    async def _get_actual_position(self):
        return self._actual_position

    async def _home(self):
        LOG.debug('start homing')
        await asyncio.sleep(1)
        await self.set_position(0 * q.mm)
        self._reset_backlash()
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
    actual_position = Quantity(q.mm, help="Actual mechanism position considering backlash")

    async def __ainit__(
        self, position=None, upper_hard_limit=None, lower_hard_limit=None, backlash=None
    ):
        await base.LinearMotor.__ainit__(self)
        await _PositionMixin.__ainit__(
            self,
            position=position,
            upper_hard_limit=upper_hard_limit,
            lower_hard_limit=lower_hard_limit,
            backlash=backlash
        )
        self._motion_velocity = 200000 * q.mm / q.s


class ContinuousLinearMotor(LinearMotor, base.ContinuousLinearMotor):

    """A continuous linear motor dummy."""

    async def __ainit__(
        self, position=None, upper_hard_limit=None, lower_hard_limit=None, backlash=None
    ):
        await base.ContinuousLinearMotor.__ainit__(self)
        await LinearMotor.__ainit__(
            self,
            position=position,
            upper_hard_limit=upper_hard_limit,
            lower_hard_limit=lower_hard_limit,
            backlash=backlash
        )

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
    actual_position = Quantity(q.deg, help="Actual mechanism position considering backlash")

    async def __ainit__(
        self,
        position=None,
        upper_hard_limit=None,
        lower_hard_limit=None,
        backlash=None
    ):
        await base.RotationMotor.__ainit__(self)
        await _PositionMixin.__ainit__(
            self,
            position=position,
            upper_hard_limit=upper_hard_limit,
            lower_hard_limit=lower_hard_limit,
            backlash=backlash
        )
        self._motion_velocity = 50000 * q.deg / q.s


class ContinuousRotationMotor(RotationMotor,
                              base.ContinuousRotationMotor):

    """A continuous rotational step motor dummy."""

    async def __ainit__(
        self,
        position=None,
        upper_hard_limit=None,
        lower_hard_limit=None,
        backlash=None
    ):
        await base.ContinuousRotationMotor.__ainit__(self)
        await RotationMotor.__ainit__(
            self,
            position=position,
            upper_hard_limit=upper_hard_limit,
            lower_hard_limit=lower_hard_limit,
            backlash=backlash
        )

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
        self.lamino_motor = await ContinuousRotationMotor(backlash=0.1 * q.deg)
        self.roll_motor = await ContinuousRotationMotor(backlash=0.1 * q.deg)
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
