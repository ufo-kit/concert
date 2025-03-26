import inspect
import time
import zmq
from concert.tests import TestCase, suppressed_logging
from concert.quantities import q
from concert.helpers import (
    convert_image,
    get_basename,
    get_state_from_awaitable,
    ImageWithMetadata,
    is_iterable,
    measure,
    memoize,
    arange,
    linspace,
    PerformanceTracker,
    CommData
)
from concert.processes.common import focus, align_rotation_axis, ProcessError
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.cameras.dummy import Camera


class TestExpects(TestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.seq = list(range(10))
        self.camera = await Camera()
        self.linear_motor = await LinearMotor()
        self.linear_motor2 = await LinearMotor()
        self.rotation_motor = await RotationMotor()

    async def test_focus_func_arguments_type_error(self):
        with self.assertRaises(TypeError):
            await focus(self.camera, self.rotation_motor)

    async def test_focus_function_arguments(self):
        await focus(self.camera, self.linear_motor)

    async def test_align_rotation_axis_func_type_error(self):
        with self.assertRaises(TypeError):
            await align_rotation_axis(self.camera, self.linear_motor).result()
        with self.assertRaises(TypeError):
            await align_rotation_axis(
                self.camera, self.rotation_motor, self.linear_motor).result()
        with self.assertRaises(TypeError):
            await align_rotation_axis(
                self.camera,
                self.rotation_motor,
                self.rotation_motor,
                self.rotation_motor,
                num_frames=[
                    10,
                    20]).result()
        with self.assertRaises(TypeError):
            await align_rotation_axis(
                self.camera,
                self.rotation_motor,
                self.rotation_motor,
                self.rotation_motor,
                num_frames=10 * q.m
            ).result()

    async def test_align_rotation_axis_function(self):
        with self.assertRaises(ProcessError):
            # Dummy camera, so no tips in noise
            await align_rotation_axis(self.camera, self.rotation_motor, x_motor=self.rotation_motor)


@suppressed_logging
def test_measure_execution():
    @measure(return_result=True)
    def sleeping():
        time.sleep(0.001)
        return 123

    result, elapsed = sleeping()
    assert(result == 123)
    assert(elapsed > 0.001 * q.s)


@suppressed_logging
def test_is_iterable():
    iterables = [(1, 2), [1, 2], {1, 2}]
    noniterables = [1, None, 1 * q.mm]

    # Standard stuff
    for item in iterables:
        assert is_iterable(item)

    # array * unit
    for item in iterables[:2]:
        item = item * q.mm
        assert is_iterable(item)

    # item * unit, item * unit
    assert is_iterable([1 * q.mm, 2 * q.mm])

    for item in noniterables:
        assert not is_iterable(item)


@suppressed_logging
def test_convert_image():
    import numpy as np
    image = np.empty((2, 8))
    image = image.view(ImageWithMetadata)
    image.metadata["rotate"] = 1
    conv = convert_image(image, False, 1)

    assert conv.shape == image.shape[::-1]
    # Rotation must be reset, otherwise second call would rotate it once more
    assert conv.metadata.get("rotate", 0) != 1


def test_performance_tracker(caplog):
    import logging
    from concert.helpers import LOG as helperlog

    with caplog.at_level(logging.DEBUG, logger=helperlog.name):
        with PerformanceTracker(loglevel=logging.DEBUG) as pt:
            time.sleep(.01)
            pt.size = 1 * q.GB
        assert pt.duration >= .01 * q.s
    assert 'size' in caplog.text


@suppressed_logging
def test_get_basename():
    assert get_basename("/foo/bar/") == "bar"
    assert get_basename("/foo/bar") == "bar"


class TestMemoize(TestCase):
    def test_ordinary_func(self):
        ran = False

        @memoize
        def func(arg):
            nonlocal ran
            ran = True
            return arg + 1

        self.assertFalse(inspect.iscoroutinefunction(func))
        self.assertEqual(func(1), 2)
        self.assertTrue(ran)
        ran = False
        self.assertEqual(func(1), 2)
        self.assertFalse(ran)

    async def test_coro_func(self):
        ran = False

        @memoize
        async def afunc(arg):
            nonlocal ran
            ran = True
            return arg + 1

        self.assertTrue(inspect.iscoroutinefunction(afunc))
        self.assertEqual(await afunc(1), 2)
        self.assertTrue(ran)
        ran = False
        self.assertEqual(await afunc(1), 2)
        self.assertFalse(ran)


class TestArangeLinspace(TestCase):
    def test_linspace_with_endpoint(self):
        num_steps = 10
        start = 0 * q.deg
        stop = (num_steps - 1) * q.deg

        x = linspace(start, stop, num_steps, endpoint=True)
        self.assertEqual(len(x), num_steps)
        for i in range(num_steps):
            self.assertEqual(x[i], float(i) * q.deg)

    def test_arange(self):
        num_steps = 10
        start = 0 * q.deg
        stop = num_steps * q.deg
        step_size = (stop - start) / num_steps

        x = arange(start, stop, step_size)
        self.assertEqual(len(x), num_steps)
        for i in range(num_steps):
            self.assertEqual(x[i], float(i) * q.deg)


class TestVarious(TestCase):
    async def test_get_state_from_awaitable(self):
        import asyncio
        from concert.coroutines.base import start

        async def _test_error(coro, exc, cancel=False):
            t = start(coro)
            if cancel:
                t.cancel()

            try:
                await t
            except exc:
                pass
            self.assertEqual(await get_state_from_awaitable(t), 'cancelled' if cancel else 'error')

        # Running
        async def long_coro():
            await asyncio.sleep(100)

        t = start(long_coro())
        await asyncio.sleep(0)
        self.assertEqual(await get_state_from_awaitable(t), 'running')
        t.cancel()

        # Normal
        async def coro():
            pass

        t = start(coro())
        await t
        self.assertEqual(await get_state_from_awaitable(t), 'standby')

        # Error
        async def error_coro():
            raise RuntimeError

        await _test_error(error_coro(), RuntimeError)

        # asyncio.CancelledError
        async def cancelled_error_coro():
            await asyncio.sleep(100)

        await _test_error(cancelled_error_coro(), asyncio.CancelledError, cancel=True)


class TestCommData(TestCase):

    def test_tcp(self):
        comms = CommData("localhost", port=1234, protocol="tcp", socket_type=zmq.PUSH, sndhwm=1234)
        self.assertEqual(comms.server_endpoint, "tcp://*:1234")
        self.assertEqual(comms.client_endpoint, "tcp://localhost:1234")
        self.assertEqual(comms.sndhwm, 1234)
        self.assertEqual(comms.socket_type, zmq.PUSH)

    def test_ipc(self):
        endpoint = "/tmp/concert/foo"
        comms = CommData(endpoint, protocol="ipc", socket_type=zmq.PUSH, sndhwm=1234)
        self.assertEqual(comms.server_endpoint, "ipc://" + endpoint)
        self.assertEqual(comms.client_endpoint, "ipc://" + endpoint)
        self.assertEqual(comms.sndhwm, 1234)
        self.assertEqual(comms.socket_type, zmq.PUSH)

    def test_wrong_protocols(self):
        with self.assertRaises(ValueError):
            comms = CommData("localhost", protocol="foo")

    def test_equality(self) -> None:
        comm1 = CommData(host="localhost", port=8991, protocol="tcp", socket_type=zmq.PUB,
                         sndhwm=-1)
        comm2 = CommData(host="localhost", port=8991, protocol="tcp", socket_type=zmq.PUB,
                         sndhwm=-1)
        comm3 = CommData(host="localhost", port=8992, protocol="tcp", socket_type=zmq.PUB,
                         sndhwm=-1)
        self.assertEqual(comm1, comm2)
        self.assertTrue(comm1 == comm2)
        self.assertNotEqual(comm1, comm3)
        self.assertFalse(comm2 == comm3)

    def test_hash_function(self) -> None:
        comm1 = CommData(host="localhost", port=8991, protocol="tcp", socket_type=zmq.PUB,
                         sndhwm=-1)
        self.assertTrue(hash(comm1) == hash(("localhost", 8991, "tcp", zmq.PUB, -1)))
        self.assertFalse(hash(comm1) == hash(("localhost", 8991, "tcp", zmq.PUSH, -1)))
