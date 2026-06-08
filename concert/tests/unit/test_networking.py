import asyncio
import numpy as np
import zmq
from concert.coroutines.base import start
from concert.helpers import ImageWithMetadata
from concert.quantities import q
from concert.networking.base import (
    ZmqSender,
    ZmqReceiver,
    ZmqBroadcaster,
    zmq_create_image_metadata,
    NetworkingError
)
from concert.tests import TestCase


CLIENT = "tcp://localhost:9999"
SERVER = "tcp://*:9999"


async def setup_broadcaster():
    sender = await ZmqSender(endpoint="tcp://*:19997")
    senders = (("tcp://*:19998", True, None), ("tcp://*:19999", False, 1))
    broadcast = await ZmqBroadcaster("tcp://localhost:19997", senders)
    # Receivers must be created after broadcast servers, otherwise first image would be lost. This
    # however does not happen at normal runtime, only during tests.
    receiver_1 = await ZmqReceiver(endpoint="tcp://localhost:19998", reliable=True)
    receiver_2 = await ZmqReceiver(endpoint="tcp://localhost:19999", reliable=False, rcvhwm=1)

    return (sender, broadcast, receiver_1, receiver_2)


class TestZmq(TestCase):
    async def asyncSetUp(self):
        super(TestZmq, self).setUp()
        self.context = zmq.asyncio.Context()
        self.sender = await ZmqSender(endpoint=SERVER)
        self.receiver = await ZmqReceiver(endpoint=CLIENT)
        self.image = np.ones((12, 16), dtype="uint16")

    async def asyncTearDown(self):
        await self.sender.close()
        await self.receiver.close()

    def test_zmq_create_image_metadata(self):
        # numpy
        meta = zmq_create_image_metadata(self.image)
        assert "dtype" in meta and meta["dtype"] == "uint16"
        assert "shape" in meta and meta["shape"] == self.image.shape

        # ImageWithMetadata
        image = ImageWithMetadata(self.image, metadata={"foo": "bar"})
        meta = zmq_create_image_metadata(image)
        assert "dtype" in meta and meta["dtype"] == "uint16"
        assert "shape" in meta and meta["shape"] == image.shape
        assert "foo" in meta and meta["foo"] == "bar"

        # End
        self.assertEqual(zmq_create_image_metadata(None), {"end": True})

    async def test_connect(self):
        # re-connection must work
        await self.receiver.connect(CLIENT)
        await self.sender.connect(SERVER)

    async def test_close(self):
        await self.sender.close()
        with self.assertRaises(NetworkingError):
            await self.sender.send_image(self.image)

    async def test_contextmanager(self):
        await self.sender.close()
        async with await ZmqSender(SERVER) as sender:
            pass

        with self.assertRaises(NetworkingError):
            await sender.send_image(self.image)

    async def test_sndhwm(self):
        await self.sender.close()
        with self.assertRaises(ValueError):
            sender = await ZmqSender(endpoint=SERVER, sndhwm=-1)

    async def test_is_message_available(self):
        self.assertFalse(await self.receiver.is_message_available(polling_timeout=10 * q.ms))
        await self.sender.send_image(self.image)
        self.assertTrue(await self.receiver.is_message_available(polling_timeout=10 * q.ms))

    async def test_send_receive(self):
        await start(self.sender.send_image(self.image))
        meta, image = await start(self.receiver.receive_image())
        np.testing.assert_equal(self.image, image)

    async def test_send_receive_json(self):
        payload = {"sample-bbox": [1, 2, 3, 4]}
        await start(self.sender.send_json(payload))
        self.assertEqual(await start(self.receiver.receive_json()), payload)

    async def test_publish_subscribe(self):
        # Make new ones
        await self.sender.close()
        await self.receiver.close()

        sender = await ZmqSender(endpoint=SERVER, reliable=False, sndhwm=1)
        receiver = await ZmqReceiver(endpoint=CLIENT, reliable=False, rcvhwm=1)
        # Start ahead to make sure we catch the image
        f = start(receiver.receive_image())
        await start(sender.send_image(self.image))
        meta, image = await f
        np.testing.assert_equal(self.image, image)

    async def test_subscribe(self):
        # Normal operation
        await start(self.sender.send_image(self.image))
        await start(self.sender.send_image(None))

        async for _ in self.receiver.subscribe(return_metadata=False):
            pass

        # Stop requested
        for i in range(10):
            await start(self.sender.send_image(self.image))
        await start(self.sender.send_image(None))

        i = 0
        async for _ in self.receiver.subscribe(return_metadata=False):
            i += 1
            f = start(self.receiver.stop())

        await f
        self.assertLessEqual(1, i)

    async def test_receiver_stop_without_subscription(self):
        await asyncio.wait_for(self.receiver.stop(), 1)
        self.assertIsNone(self.receiver._stopped)

    async def test_receiver_stop_requests_blocked_subscription(self):
        async def consume():
            async for _ in self.receiver.subscribe():
                pass

        subscription = start(consume())
        await asyncio.sleep(0)
        await asyncio.wait_for(self.receiver.stop(), 1)
        await asyncio.wait_for(subscription, 1)

        self.assertTrue(self.receiver._stopped.is_set())
        self.assertFalse(self.receiver._request_stop)

    async def test_receiver_can_subscribe_after_stop(self):
        async def consume():
            async for image in self.receiver.subscribe():
                return image

        first = start(consume())
        await asyncio.sleep(0)
        await asyncio.wait_for(self.receiver.stop(), 1)
        await asyncio.wait_for(first, 1)

        second = start(consume())
        await self.sender.send_image(self.image)
        image = await asyncio.wait_for(second, 1)
        np.testing.assert_equal(image, self.image)

    async def test_broadcast_immediate_shutdown(self):
        sender, broadcast, receiver_1, receiver_2 = await setup_broadcaster()
        f = start(broadcast.serve())
        await broadcast.shutdown()
        await asyncio.wait_for(f, 1)

    async def test_broadcast(self):
        sender, broadcast, receiver_1, receiver_2 = await setup_broadcaster()
        f = start(broadcast.serve())
        # Start ahead to make sure we catch the image
        f_sub = start(receiver_2.receive_image())

        await start(sender.send_image(self.image))
        meta, image = await start(receiver_1.receive_image())
        np.testing.assert_equal(self.image, image)
        meta, image = await f_sub
        np.testing.assert_equal(self.image, image)
        await broadcast.shutdown()
        await f

    async def test_receiver_timeout(self):
        receiver = await ZmqReceiver(endpoint=CLIENT, reliable=True, timeout=0.3 * q.s)
        with self.assertRaises(TimeoutError):
            await receiver.receive_image()

    async def test_sender_timeout(self):
        sender = await ZmqSender(endpoint="tcp://*:19999", timeout=0.3 * q.s)
        with self.assertRaises(TimeoutError):
            await sender.send_image(self.image)
