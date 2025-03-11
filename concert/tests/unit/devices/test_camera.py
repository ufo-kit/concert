from datetime import datetime
from unittest import mock
import numpy as np
import zmq
from concert.tests import TestCase
from concert.quantities import q
from concert.devices.cameras.dummy import Camera, BufferedCamera
from concert.devices.cameras.pco import Timestamp, TimestampError
from concert.helpers import CommData, convert_image, ImageWithMetadata
from concert.networking.base import ZmqSender, ZmqReceiver


class TestDummyCamera(TestCase):

    async def asyncSetUp(self):
        await super(TestDummyCamera, self).asyncSetUp()
        self.background = np.ones((256, 256), dtype=np.uint16)
        self.camera = await Camera(background=self.background)

    async def test_grab(self):
        async with self.camera.recording():
            frame = await self.camera.grab()
        self.assertIsNotNone(frame)

    def test_trigger_source(self):
        self.camera.trigger_source = self.camera.trigger_sources.EXTERNAL
        self.assertEqual(self.camera.trigger_source, 'EXTERNAL')

    def test_roi(self):
        self.roi_x0 = self.roi_y0 = self.roi_width = self.roi_height = 10 * q.dimensionless
        self.assertEqual(self.roi_x0, 10 * q.dimensionless)
        self.assertEqual(self.roi_y0, 10 * q.dimensionless)
        self.assertEqual(self.roi_width, 10 * q.dimensionless)
        self.assertEqual(self.roi_height, 10 * q.dimensionless)

    def test_has_sensor_dims(self):
        self.assertTrue(hasattr(self.camera, "sensor_pixel_width"))
        self.assertTrue(hasattr(self.camera, "sensor_pixel_height"))

    async def test_buffered_camera(self):
        camera = await BufferedCamera()
        i = 0

        await camera.start_readout()
        try:
            producer = camera.readout_buffer()
            print(producer)
            async for item in producer:
                i += 1
        finally:
            await camera.stop_readout()

        self.assertEqual(i, 3)

    async def test_context_manager(self):
        camera = await Camera()

        async with camera.recording():
            self.assertEqual(await camera.get_state(), 'recording')
            await camera.grab()

        self.assertEqual(await camera.get_state(), 'standby')

    async def test_stream(self):
        async def check(producer):
            async for image in producer:
                check.ok = True
                await self.camera.stop_recording()
                break
        check.ok = False

        await check(self.camera.stream())
        self.assertTrue(check.ok)

    async def test_grab_convert(self):
        async def grab():
            return np.mgrid[:5, :5][1]

        self.camera._grab_real = grab

        for mirrored in (True, False):
            for rotated in (0, 1, 2, 3):
                self.camera.set_mirror(mirrored)
                self.camera.set_rotate(rotated)
                async with self.camera.recording():
                    image = await self.camera.grab()
                np.testing.assert_equal(image, convert_image(await grab(), mirrored, rotated))

    async def test_grab_and_send_convert(self):
        async def grab():
            return np.mgrid[:5, :5][1]

        self.camera._grab_real = grab

        i = 0
        for mirrored in (True, False):
            for rotated in (0, 1, 2, 3):
                await self.camera.unregister_all()
                await self.camera.register_endpoint(CommData("localhost", 8991+i, "tcp", zmq.PUSH, 0))
                receiver = ZmqReceiver(endpoint=f"tcp://localhost:{8991+i}")

                self.camera.set_mirror(mirrored)
                self.camera.set_rotate(rotated)
                async with self.camera.recording():
                    await self.camera.grab_send(1)
                    (metadata, image) = await receiver.receive_image()
                receiver.close()
                i += 1
                np.testing.assert_equal(image, convert_image(await grab(), mirrored, rotated))


    async def test_simulate(self):
        async with self.camera.recording():
            self.assertTrue(np.any(self.background - await self.camera.grab()))

        camera = await Camera(background=self.background, simulate=False)
        async with camera.recording():
            np.testing.assert_equal(self.background, await camera.grab())

#    async def test_endpoint_registration(self) -> None:
#
#        async def mock_register_endpoint(self, endpoint: CommData) -> None:
#            if endpoint in self._senders:
#                raise ValueError("zmq endpoint already in list")
#            self._senders[endpoint] = mock.MagicMock(
#                endpoint.server_endpoint,
#                reliable=endpoint.socket_type == zmq.PUSH,
#                sndhwm=endpoint.sndhwm
#            )
#
#        comm1 = CommData(host="localhost", port=8991, protocol="tcp", socket_type=zmq.PUSH,
#                         sndhwm=0)
#        comm2 = CommData(host="localhost", port=8991, protocol="tcp", socket_type=zmq.PUSH,
#                         sndhwm=0)
#        comm3 = CommData(host="localhost", port=8992, protocol="tcp", socket_type=zmq.PUSH,
#                         sndhwm=0)
#        self.assertTrue(comm1 == comm2)
#        fqn_func = "concert.devices.cameras.base.Camera.register_endpoint"
#        with mock.patch(fqn_func, wraps=mock_register_endpoint) as mre:
#            try:
#                await self.camera.register_endpoint(endpoint=comm1)
#            except Exception:
#                self.fail("first endpoint registration must not fail")
#            with self.assertRaises(ValueError):
#                await self.camera.register_endpoint(endpoint=comm2)
#            try:
#                await self.camera.register_endpoint(endpoint=comm3)
#            except Exception:
#                self.fail("new endpoint registration must not fail")
#            try:
#                await self.camera.unregister_endpoint(endpoint=comm1)
#            except Exception:
#                self.fail("removing a registered endpoint must not fail")


class TestPCOTimeStamp(TestCase):
    def test_valid(self):
        image = np.empty((1, 14), dtype=np.uint16)
        image[0] = np.array([0, 0, 0, 1, 32, 20, 8, 1, 20, 80, 69, 147, 40, 9], dtype=np.uint16)
        stamp = Timestamp(image)
        date_time = datetime(2014, 8, 1, 14, 50, 45, 932809)

        self.assertEqual(stamp.time, date_time)
        self.assertEqual(stamp.number, 1)

    def test_bad_input(self):
        image = np.empty((1,))
        with self.assertRaises(TypeError):
            Timestamp(image)

        image = np.empty((1,), dtype=np.uint16)
        with self.assertRaises(ValueError):
            Timestamp(image)

    def test_bad_data(self):
        image = np.empty((1, 14), dtype=np.uint16)
        image[0] = np.array([0, 0, 0, 1, 32, 20, 2**15, 1, 20, 80, 69, 147, 40, 9], dtype=np.uint16)
        with self.assertRaises(TimestampError):
            Timestamp(image)
