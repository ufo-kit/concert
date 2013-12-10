import numpy as np
from concert.coroutines import coroutine, broadcast, inject
from concert.coroutines.filters import flat_correct, average_images, sinograms, downsize
from concert.coroutines.sinks import null, Result
from concert.tests import TestCase


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
        frame_producer(average_images(3, self.consume()))
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
