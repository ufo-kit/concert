import asyncio
import numpy as np
from concert.coroutines.base import (async_generate, broadcast, feed_queue, run_in_executor,
                                     run_in_loop, start, wait_until, WaitError)
from concert.coroutines.filters import (absorptivity, flat_correct, average_images,
                                        downsize, stall, Timer)
from concert.coroutines.sinks import null, Result, Accumulate
from concert.quantities import q
from concert.tests import assert_almost_equal, TestCase


async def produce_frames(num_frames=3, shape=(2, 2)):
    for i in range(num_frames):
        yield np.ones(shape=shape) * (i + 1)


class TestCoroutines(TestCase):

    def setUp(self):
        super(TestCoroutines, self).setUp()
        self.data = None
        self.iteration = 0

    async def produce(self, num_items=5):
        for i in range(num_items):
            yield i
            self.iteration += 1

    async def consume(self, producer):
        async for data in producer:
            self.data = data

    async def test_broadcast(self):

        async def consume(producer):
            nonlocal data
            async for item in producer:
                data = item

        data = None
        coros = broadcast(self.produce(), self.consume, consume)
        await asyncio.gather(*coros)
        self.assertEqual(self.data, 4)
        self.assertEqual(data, 4)

    async def test_broadcast_consumer_break(self):

        async def consume(producer):
            nonlocal data
            async for item in producer:
                data = item
                break

        data = None
        coros = broadcast(self.produce(), self.consume, consume)
        await asyncio.gather(*coros)
        # First gets the whole stream, the breaking one just the item before break
        self.assertEqual(self.data, 4)
        self.assertEqual(data, 0)

    async def test_broadcast_cancel(self):
        produced = asyncio.Event()

        async def produce():
            for i in range(3):
                yield i
                produced.set()
                # Allow future.cancel() to kick in
                await asyncio.sleep(0)

        coros = broadcast(produce(), self.consume)
        future = asyncio.gather(*coros)
        await produced.wait()
        future.cancel()
        await asyncio.wait([future])
        self.assertEqual(self.data, 0)

    async def test_feed_queue(self):
        produced = asyncio.Event()
        item = None

        async def produce():
            for i in range(3):
                yield i
                produced.set()
                # Allow future.cancel() to kick in
                await asyncio.sleep(0)

        def blocking_consumer(queue, check_cancelled=False):
            nonlocal item
            while True:
                item = queue.get()
                queue.task_done()
                if item.data is None:
                    break

        future = start(feed_queue(produce(), blocking_consumer))
        await future
        # Normal operation, priority with counter must arrive with None as data
        self.assertEqual(item.priority, 4)
        self.assertEqual(item.data, None)
        produced.clear()

        future = start(feed_queue(produce(), blocking_consumer))
        await produced.wait()
        future.cancel()
        # TODO: for some reason simple "await future" hangs forever, perhaps a bug in the test
        # suite?
        await asyncio.wait([future])
        # On cancel, the highest priority (0) must arrive with None as data
        self.assertTrue(future.cancelled())
        self.assertEqual(item.priority, 0)
        self.assertEqual(item.data, None)

    async def test_wait_until(self):
        ran = False

        async def do_slowly():
            nonlocal ran
            await asyncio.sleep(0.1)
            ran = True

        async def condition():
            return ran

        await asyncio.gather(wait_until(condition), do_slowly())
        self.assertTrue(ran)
        ran = False

        await asyncio.gather(wait_until(condition, timeout=10 * q.s), do_slowly())
        self.assertTrue(ran)
        ran = False

        with self.assertRaises(WaitError):
            await asyncio.gather(wait_until(condition, timeout=0.001 * q.s), do_slowly())

    def test_run_in_loop(self):
        async def consume():
            return 1

        self.assertEqual(run_in_loop(consume()), 1)

    async def test_run_in_executor(self):
        def blocking(arg):
            return arg

        self.assertEqual(await run_in_executor(blocking, 'foo'), 'foo')

    async def test_start(self):
        future = start(null(self.produce()))
        self.assertTrue(isinstance(future, asyncio.Future))
        await asyncio.sleep(.1)
        self.assertTrue(self.iteration != 0)

    async def test_null(self):
        await null(self.produce())
        self.assertEqual(5, self.iteration)

    async def test_averager(self):
        await self.consume(average_images(produce_frames()))
        truth = np.ones((2, 2)) * 2
        np.testing.assert_almost_equal(self.data, truth)

    async def test_flat_correct(self):
        shape = (2, 2)
        dark = np.ones(shape)
        flat = np.ones(shape) * 10
        truth_base = np.ones(shape)

        async def check(producer):
            i = 1
            async for frame in producer:
                value = (i - 1) / 9.0
                np.testing.assert_almost_equal(frame, truth_base * value)
                i += 1

        await check(flat_correct(flat, produce_frames(), dark=dark))

    async def test_absorptivity(self):
        truth_base = np.ones((2, 2))

        async def check(producer):
            i = 1
            async for frame in producer:
                np.testing.assert_almost_equal(frame, -np.log(truth_base * i))
                i += 1
        await check(absorptivity(produce_frames()))

    async def test_result_object(self):
        result = Result()
        await result(self.produce())
        self.assertEqual(result.result, 4)

    async def test_downsize(self):
        acc = Accumulate()
        await acc(downsize(produce_frames(num_frames=10, shape=(10, 10)),
                           x_slice=(2, 6, 2), y_slice=(3, 10, 3), z_slice=(2, 8, 3)))
        result = np.array(acc.items)
        ones = np.ones(shape=(3, 2))

        self.assertEqual(result.shape, (2, 3, 2))
        np.testing.assert_almost_equal(result[0], ones * 3)
        np.testing.assert_almost_equal(result[1], ones * 6)

    async def test_stall(self):
        # Simple case, no modulo
        acc = Accumulate()
        await acc(stall(self.produce(num_items=10), per_shot=5))
        self.assertEqual(acc.items, [4, 9])

        # More iterations than total items
        await acc(stall(self.produce(num_items=26), per_shot=8, flush_at=13))
        self.assertEqual(acc.items, [7, 12, 20, 25])

    async def test_accumulate(self):
        shape = (4, 4, 4)
        dtype = np.ushort
        data = np.ones(shape, dtype=dtype)
        double = np.concatenate((data, data))

        # List
        # Append
        accumulate = Accumulate(reset_on_call=False)
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, data)
        self.assertTrue(isinstance(accumulate.items, list))
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, double)
        accumulate.reset()
        self.assertEqual(len(accumulate.items), 0)

        # Reset
        accumulate = Accumulate(reset_on_call=True)
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, data)
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, data)

        # ndarray
        # Append
        accumulate = Accumulate(shape=shape, dtype=dtype, reset_on_call=False)
        self.assertEqual(accumulate.items.dtype, data.dtype)
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, data)
        self.assertTrue(isinstance(accumulate.items, np.ndarray))
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, double)
        accumulate.reset()
        self.assertEqual(accumulate.items.shape, (0,) + shape[1:])

        # Reset
        accumulate = Accumulate(reset_on_call=True)
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, data)
        await accumulate(async_generate(data))
        np.testing.assert_equal(accumulate.items, data)


class TestTimer(TestCase):

    def setUp(self):
        self.timer = Timer()

    async def asyncSetUp(self):
        await null(self.timer(async_generate([1, 2, 3])))

    def test_durations(self):
        self.assertEqual(len(self.timer.durations), 3)

    def test_duration(self):
        """Test the sum of durations."""
        assert_almost_equal(self.timer.duration, sum(self.timer.durations))

    def test_mean(self):
        assert_almost_equal(self.timer.mean, self.timer.duration / len(self.timer.durations))

    def test_reset(self):
        self.timer.reset()
        self.assertEqual(len(self.timer.durations), 0)
