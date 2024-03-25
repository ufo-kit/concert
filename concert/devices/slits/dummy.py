from concert.devices.motors.dummy import LinearMotor
from concert.devices.slits.base import BladeSlits as BaseBladeSlits
from concert.quantities import q


class BladeSlits(BaseBladeSlits):
    async def __ainit__(self):
        await super().__ainit__(motor_top=await LinearMotor(position=1 * q.mm),
                                motor_bottom=await LinearMotor(position=1 * q.mm),
                                motor_left=await LinearMotor(position=1 * q.mm),
                                motor_right=await LinearMotor(position=1 * q.mm),
                                top_center=0 * q.mm,
                                bottom_center=0 * q.mm,
                                left_center=0 * q.mm,
                                right_center=0 * q.mm,
                                top_positive_out=True,
                                bottom_positive_out=True,
                                left_positive_out=True,
                                right_positive_out=True
                                )
