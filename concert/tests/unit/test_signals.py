import asyncio

from concert.tests import TestCase
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor


class TestSignal(TestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.motor1 = await LinearMotor()
        self.motor2 = await LinearMotor()

    async def test_value_set_connect_and_disconnect(self):
        await self.motor1.set_position(10 * q.mm)
        await self.motor2.set_position(20 * q.mm)
        self.assertEqual(await self.motor1.get_position(), 10 * q.mm)
        self.assertEqual(await self.motor2.get_position(), 20 * q.mm)

        # Connect setting of motor1 to also set motor2 to the same position
        self.motor1['position'].value_set.connect(self.motor2, self.motor2.set_position)
        await self.motor1.set_position(25 * q.mm)

        # We made the pseudo motors to 'realistic'
        await asyncio.sleep(0.1)
        while self.motor2.get_state() == "moving":
            await asyncio.sleep(0.01)

        self.assertEqual(await self.motor1.get_position(), 25 * q.mm)
        self.assertEqual(await self.motor2.get_position(), 25 * q.mm)

        # Make sure that disconnecting the signal works
        self.motor1['position'].value_set.disconnect(self.motor2, self.motor2.set_position)
        await self.motor1.set_position(10 * q.mm)
        await self.motor2.set_position(20 * q.mm)

        self.assertEqual(await self.motor1.get_position(), 10 * q.mm)
        self.assertEqual(await self.motor2.get_position(), 20 * q.mm)




