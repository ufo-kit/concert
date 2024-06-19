from concert.base import Quantity, Parameter, State, check
from concert.devices.base import Device
from concert.quantities import q
from concert.devices.motors.base import LinearMotor


class Slits(Device):
    top = Quantity(q.mm, help='Top slit position',
                   check=check(source='standby', state_name='vertical_state'))
    bottom = Quantity(q.mm, help='Bottom slit position',
                      check=check(source='standby', state_name='vertical_state'))
    left = Quantity(q.mm, help='Left slit position',
                    check=check(source='standby', state_name='horizontal_state'))
    right = Quantity(q.mm, help='Right slit position',
                     check=check(source='standby', state_name='horizontal_state'))
    vertical_gap = Quantity(q.mm, help='Vertical gap',
                            check=check(source='standby', state_name='vertical_state'))
    horizontal_gap = Quantity(q.mm, help='Horizontal gap',
                              check=check(source='standby', state_name='horizontal_state'))
    vertical_offset = Quantity(q.mm, help='Vertical offset',
                               check=check(source='standby', state_name='vertical_state'))
    horizontal_offset = Quantity(q.mm, help='Horizontal offset',
                                 check=check(source='standby', state_name='horizontal_state'))
    horizontal_state = State(help='Horizontal state')
    vertical_state = State(help='Vertical state')
    area = Quantity(q.mm ** 2, help='Slit area')

    async def __ainit__(self):
        await super().__ainit__()

    async def _get_area(self):
        return await self.get_horizontal_gap() * await self.get_vertical_gap()


class BladeSlits(Slits):
    """
    Slit that is composed of four blades. The blades can be moved independently.
    Offsets and openings (gaps) are calculated from the blade positions.
    """
    # noinspection PyMethodOverriding
    async def __ainit__(self,
                        motor_top: LinearMotor,
                        motor_bottom: LinearMotor,
                        motor_left: LinearMotor,
                        motor_right: LinearMotor,
                        top_center=0 * q.mm,
                        bottom_center=0 * q.mm,
                        left_center=0 * q.mm,
                        right_center=0 * q.mm,
                        top_positive_out=True,
                        bottom_positive_out=True,
                        left_positive_out=True,
                        right_positive_out=True):
        """
        Initialize the BladeSlits.

        :param motor_top: Motor for the top blade.
        :param motor_bottom: Motor for the bottom blade.
        :param motor_left: Motor for the left blade.
        :param motor_right: Motor for the right blade.
        :param top_center: Center position for the top blade.
        :param bottom_center: Center position for the bottom blade.
        :param left_center: Center position for the left blade.
        :param right_center: Center position for the right blade.
        :param top_positive_out: True if the top blade moves outwards for positive positions.
        :param bottom_positive_out: True if the bottom blade moves outwards for positive positions.
        :param left_positive_out: True if the left blade moves outwards for positive positions.
        :param right_positive_out: True if the right blade moves outwards for positive positions.

        """
        self._motor_top = motor_top
        self._motor_bottom = motor_bottom
        self._motor_left = motor_left
        self._motor_right = motor_right

        self._top_center = top_center
        self._bottom_center = bottom_center
        self._left_center = left_center
        self._right_center = right_center

        self._top_positive_out = top_positive_out
        self._bottom_positive_out = bottom_positive_out
        self._left_positive_out = left_positive_out
        self._right_positive_out = right_positive_out

        await super().__ainit__()

    async def _get_vertical_state(self):
        top_state = await self._motor_top.get_state()
        bottom_state = await self._motor_bottom.get_state()
        if 'moving' in [top_state, bottom_state]:
            return 'moving'
        if 'hard-limit' in [top_state, bottom_state]:
            return 'hard-limit'
        return 'standby'

    async def _get_horizontal_state(self):
        left_state = await self._motor_left.get_state()
        right_state = await self._motor_right.get_state()
        if left_state == 'moving' or right_state == 'moving':
            return 'moving'
        if left_state == 'hard-limit' or right_state == 'hard-limit':
            return 'hard-limit'
        return 'standby'

    async def _get_top(self):
        return (await self._motor_top.get_position() - self._top_center) * (1 if self._top_positive_out else -1)

    async def _set_top(self, value):
        await self._motor_top.set_position(value * (1 if self._top_positive_out else -1) + self._top_center)

    async def _get_bottom(self):
        return (await self._motor_bottom.get_position() - self._bottom_center) * (
            1 if self._bottom_positive_out else -1)

    async def _set_bottom(self, value):
        await self._motor_bottom.set_position(value * (1 if self._bottom_positive_out else -1) + self._bottom_center)

    async def _get_left(self):
        return (await self._motor_left.get_position() - self._left_center) * (1 if self._left_positive_out else -1)

    async def _set_left(self, value):
        await self._motor_left.set_position(value * (1 if self._left_positive_out else -1) + self._left_center)

    async def _get_right(self):
        return (await self._motor_right.get_position() - self._right_center) * (1 if self._right_positive_out else -1)

    async def _set_right(self, value):
        await self._motor_right.set_position(value * (1 if self._right_positive_out else -1) + self._right_center)

    async def _get_horizontal_gap(self):
        return await self._motor_right.get_position() + await self._motor_left.get_position()

    async def _set_horizontal_gap(self, value):
        offset = await self.get_horizontal_offset()
        await self._motor_right.set_position(value / 2 + offset)
        await self._motor_left.set_position(value / 2 + offset)

    async def _get_vertical_gap(self):
        return await self._motor_top.get_position() + await self._motor_bottom.get_position()

    async def _set_vertical_gap(self, value):
        offset = await self.get_vertical_offset()
        await self._motor_top.set_position(value / 2 + offset)
        await self._motor_bottom.set_position(value / 2 + offset)

    async def _get_horizontal_offset(self):
        return (await self._motor_right.get_position() - await self._motor_left.get_position()) / 2

    async def _get_vertical_offset(self):
        return (await self._motor_top.get_position() - await self._motor_bottom.get_position()) / 2


class GapSlits(Slits):
    """
    Slits that are composed of two sets of blades. The physical motors move the opening and the offset.
    """
    # TODO
    pass
