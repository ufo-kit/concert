import numpy as np
from concert.coroutines.base import coroutine, broadcast, inject
from concert.coroutines.filters import (absorptivity, backproject, flat_correct, average_images,
                                        queue, sinograms, downsize, stall, PickSlice, Timer)
from concert.coroutines.sinks import null, Result, Accumulate
from concert.tests import assert_almost_equal, TestCase


def generator():
    for i in range(5):
        yield i


def frame_producer(consumer, num_frames=3, shape=(2, 2)):
    for i in range(num_frames):
        consumer.send(np.ones(shape=shape) * (i + 1))


class TestCoroutines(TestCase):

    def setUp(self):
        super(TestCoroutines, self).setUp()
        self.data = None
        self.data_2 = None
        self.stack = []
        self.iteration = 0

    def producer(self, consumer, num_items=5):
        for i in range(num_items):
            consumer.send(i)
            self.iteration += 1

    @coroutine
    def consume(self):
        while True:
            self.data = yield

    @coroutine
    def consume_2(self):
        while True:
            self.data_2 = yield

    @coroutine
    def stack_consume(self):
        while True:
            item = yield
            self.stack.append(item)

    def test_broadcast(self):
        self.producer(broadcast(self.consume(), self.consume_2()))
        self.assertEqual(self.data, 4)
        self.assertEqual(self.data_2, 4)

    def test_sinogenerator(self):
        n = 4

        result = Result()
        frame_producer(sinograms(n, result()), num_frames=8, shape=(4, 4))

        sinos = result.result
        # One sinogram
        one = np.tile(np.array([5, 6, 7, 8])[:, np.newaxis], [1, n])
        np.testing.assert_almost_equal(sinos, np.tile(one, [n, 1, 1]))

    def test_injection(self):
        inject(generator(), self.consume())
        self.assertEqual(self.data, 4)

    def test_null(self):
        self.producer(null())
        self.assertEquals(5, self.iteration)

    def test_averager(self):
        frame_producer(average_images(self.consume()))
        truth = np.ones((2, 2)) * 2
        np.testing.assert_almost_equal(self.data, truth)

    def test_flat_correct(self):
        shape = (2, 2)
        dark = np.ones(shape)
        flat = np.ones(shape) * 10
        truth_base = np.ones(shape)

        @coroutine
        def check():
            i = 1
            while True:
                frame = yield
                value = (i - 1) / 9.0
                np.testing.assert_almost_equal(frame, truth_base * value)
                i += 1

        frame_producer(flat_correct(flat, check(), dark=dark))

    def test_absorptivity(self):
        truth_base = np.ones((2, 2))

        @coroutine
        def check():
            i = 1
            while True:
                frame = yield
                np.testing.assert_almost_equal(frame, -np.log(truth_base * i))
                i += 1
        frame_producer(absorptivity(check()))

    def test_result_object(self):
        result = Result()
        self.producer(result())
        self.assertEqual(result.result, 4)

    def test_downsize(self):
        frame_producer(downsize(self.stack_consume(), x_slice=(2, 6, 2), y_slice=(3, 10, 3),
                                z_slice=(2, 8, 3)), num_frames=10, shape=(10, 10))
        result = np.array(self.stack)
        ones = np.ones(shape=(3, 2))

        self.assertEqual(result.shape, (2, 3, 2))
        np.testing.assert_almost_equal(result[0], ones * 3)
        np.testing.assert_almost_equal(result[1], ones * 6)

    def test_stall(self):
        # Simple case, no modulo
        self.producer(stall(self.stack_consume(), per_shot=5), num_items=10)
        self.assertEqual(self.stack, [4, 9])

        # More iterations than total items
        self.stack = []
        self.producer(stall(self.stack_consume(), per_shot=8, flush_at=13), num_items=26)
        self.assertEqual(self.stack, [7, 12, 20, 25])

    def test_slicepick(self):
        def produce_volume(consumer):
            vol = np.ones((2, 2, 2))
            vol[1] *= 2
            consumer.send(vol)

        pick = PickSlice(0)

        produce_volume(pick(self.consume()))
        np.testing.assert_almost_equal(self.data, np.ones((2, 2)))

        pick.index = 1
        produce_volume(pick(self.consume()))
        np.testing.assert_almost_equal(self.data, np.ones((2, 2)) * 2)

    def test_queue(self):
        frame_producer(queue(null()))

    def test_backproject(self):
        frame_producer(backproject(1, null()))

    def test_accumulate(self):
        accumulate = Accumulate()
        inject(generator(), accumulate())
        self.assertEqual(accumulate.items, range(5))


class TestTimer(TestCase):

    def setUp(self):
        self.timer = Timer()
        inject([1, 2, 3], self.timer(null()))

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
