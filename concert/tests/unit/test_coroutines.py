import numpy as np
from concert.coroutines import coroutine, broadcast, inject
from concert.coroutines.filters import (flat_correct,
                                        average_images,
                                        make_sinograms)
from concert.coroutines.sinks import null, Result
from concert.tests import TestCase


def generator():
    for i in range(5):
        yield i


def frame_producer(consumer, num_frames=3):
    for i in range(num_frames):
        consumer.send(np.ones((2, 2)) * (i + 1))


class TestCoroutines(TestCase):

    def setUp(self):
        super(TestCoroutines, self).setUp()
        self.data = None
        self.data_2 = None
        self.iteration = 0

    def producer(self, consumer):
        for i in range(5):
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

    def test_broadcast(self):
        self.producer(broadcast(self.consume(), self.consume_2()))
        self.assertEqual(self.data, 4)
        self.assertEqual(self.data_2, 4)

    def test_sinogenerator(self):
        n = 4
        ground_truth = np.zeros((n, n, n))

        def image_producer(consumer):
            for i in range(n):
                ground_truth[i] = np.arange(n ** 2).reshape(n, n) * (i + 1)
                consumer.send(ground_truth[i])

        result = Result()
        image_producer(make_sinograms(n, result()))

        sinograms = result.result
        np.testing.assert_almost_equal(sinograms,
                                       ground_truth.transpose(1, 0, 2))

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
