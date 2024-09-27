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
from concert.tests import assert_almost_equal, TestCase


CLIENT = "tcp://localhost:9999"
SERVER = "tcp://*:9999"


def setup_broadcaster():
    sender = ZmqSender(endpoint="tcp://*:19997")
    senders = (("tcp://*:19998", True, None), ("tcp://*:19999", False, 1))
    broadcast = ZmqBroadcaster("tcp://localhost:19997", senders)
    # Receivers must be created after broadcast servers, otherwise first image would be lost. This
    # however does not happen at normal runtime, only during tests.
    receiver_1 = ZmqReceiver(endpoint="tcp://localhost:19998", reliable=True)
    receiver_2 = ZmqReceiver(endpoint="tcp://localhost:19999", reliable=False, rcvhwm=1)

    return (sender, broadcast, receiver_1, receiver_2)


class TestZmq(TestCase):
    def setUp(self):
        super(TestZmq, self).setUp()
        self.context = zmq.asyncio.Context()
        self.sender = ZmqSender(endpoint=SERVER)
        self.receiver = ZmqReceiver(endpoint=CLIENT)
        self.image = np.ones((12, 16), dtype="uint16")

    def tearDown(self):
        self.sender.close()
        self.receiver.close()

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

    def test_connect(self):
        # re-connection must work
        self.receiver.connect(CLIENT)
        self.sender.connect(SERVER)

    async def test_close(self):
        self.sender.close()
        with self.assertRaises(NetworkingError):
            await self.sender.send_image(self.image)

    async def test_contextmanager(self):
        self.sender.close()
        with ZmqSender(SERVER) as sender:
            pass

        with self.assertRaises(NetworkingError):
            await sender.send_image(self.image)

    def test_sndhwm(self):
        self.sender.close()
        with self.assertRaises(ValueError):
            sender = ZmqSender(endpoint=SERVER, sndhwm=-1)

    async def test_is_message_available(self):
        self.assertFalse(await self.receiver.is_message_available(polling_timeout=10 * q.ms))
        await self.sender.send_image(self.image)
        self.assertTrue(await self.receiver.is_message_available(polling_timeout=10 * q.ms))

    async def test_send_receive(self):
        await start(self.sender.send_image(self.image))
        meta, image = await start(self.receiver.receive_image())
        np.testing.assert_equal(self.image, image)

    async def test_publish_subscribe(self):
        # Make new ones
        self.sender.close()
        self.receiver.close()

        sender = ZmqSender(endpoint=SERVER, reliable=False, sndhwm=1)
        receiver = ZmqReceiver(endpoint=CLIENT, reliable=False, rcvhwm=1)
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
        await start(self.sender.send_image(self.image))
        await start(self.sender.send_image(self.image))
        await start(self.sender.send_image(None))

        i = 0
        async for _ in self.receiver.subscribe(return_metadata=False):
            i += 1
            self.receiver.stop()

        self.assertEqual(i, 1)


    async def test_broadcast_immediate_shutdown(self):
        sender, broadcast, receiver_1, receiver_2 = setup_broadcaster()
        f = start(broadcast.serve())
        await broadcast.shutdown()
        await asyncio.wait_for(f, 1)

    async def test_broadcast(self):
        sender, broadcast, receiver_1, receiver_2 = setup_broadcaster()
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
        receiver = ZmqReceiver(endpoint=CLIENT, reliable=True, timeout=0.3 * q.s)
        with self.assertRaises(TimeoutError):
            await receiver.receive_image()

    async def test_sender_timeout(self):
        sender = ZmqSender(endpoint="tcp://*:19999", timeout=0.3 * q.s)
        with self.assertRaises(TimeoutError):
            await sender.send_image(self.image)
