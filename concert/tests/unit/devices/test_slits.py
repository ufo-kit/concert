import asyncio

from concert.devices.slits.dummy import BladeSlits
from concert.tests import TestCase
from concert.quantities import q


class TestBladeSlits(TestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.slits = await BladeSlits()

    async def test_blades(self):
        await self.slits.set_top(2 * q.mm)
        self.assertEqual(2 * q.mm, await self.slits.get_top())

        await self.slits.set_top(2 * q.mm)
        self.assertEqual(2 * q.mm, await self.slits.get_top())

        await self.slits.set_bottom(2 * q.mm)
        self.assertEqual(2 * q.mm, await self.slits.get_bottom())

        await self.slits.set_left(2 * q.mm)
        self.assertEqual(2 * q.mm, await self.slits.get_left())

    async def test_vertical_gap_offsets(self):
        await self.slits.set_top(2 * q.mm)
        await self.slits.set_bottom(2 * q.mm)
        self.assertEqual(0 * q.mm, await self.slits.get_vertical_offset())
        self.assertEqual(4 * q.mm, await self.slits.get_vertical_gap())

        await self.slits.set_top(2 * q.mm)
        await self.slits.set_bottom(0 * q.mm)
        self.assertEqual(1 * q.mm, await self.slits.get_vertical_offset())
        self.assertEqual(2 * q.mm, await self.slits.get_vertical_gap())

    async def test_horizontal_gap_offset(self):
        await self.slits.set_left(2 * q.mm)
        await self.slits.set_right(2 * q.mm)
        self.assertEqual(0 * q.mm, await self.slits.get_horizontal_offset())
        self.assertEqual(4 * q.mm, await self.slits.get_horizontal_gap())

        await self.slits.set_left(2 * q.mm)
        await self.slits.set_right(0 * q.mm)
        self.assertEqual(-1 * q.mm, await self.slits.get_horizontal_offset())
        self.assertEqual(2 * q.mm, await self.slits.get_horizontal_gap())

    async def test_states(self):
        await self.slits.set_top(0 * q.mm)
        await self.slits.set_bottom(0 * q.mm)
        await self.slits.set_left(0 * q.mm)
        await self.slits.set_right(0 * q.mm)

        self.slits._motor_top.set_motion_velocity(1 * q.mm / q.s)
        self.slits._motor_bottom.set_motion_velocity(1 * q.mm / q.s)
        self.slits._motor_left.set_motion_velocity(1 * q.mm / q.s)
        self.slits._motor_right.set_motion_velocity(1 * q.mm / q.s)

        self.assertEqual('standby', await self.slits.get_vertical_state())
        self.assertEqual('standby', await self.slits.get_horizontal_state())

        f = self.slits.set_top(0.1 * q.mm)
        await asyncio.sleep(0.01)
        self.assertEqual('moving', await self.slits.get_vertical_state())
        self.assertEqual('standby', await self.slits.get_horizontal_state())
        await f

        f = self.slits.set_bottom(0.1 * q.mm)
        await asyncio.sleep(0.01)
        self.assertEqual('moving', await self.slits.get_vertical_state())
        self.assertEqual('standby', await self.slits.get_horizontal_state())
        await f

        f = self.slits.set_left(0.1 * q.mm)
        await asyncio.sleep(0.01)
        self.assertEqual('moving', await self.slits.get_horizontal_state())
        self.assertEqual('standby', await self.slits.get_vertical_state())
        await f

        f = self.slits.set_right(0.1 * q.mm)
        await asyncio.sleep(0.01)
        self.assertEqual('moving', await self.slits.get_horizontal_state())
        self.assertEqual('standby', await self.slits.get_vertical_state())
        await f
