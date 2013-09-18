from concert.tests.base import suppressed_logging, ConcertTest
from concert.base import coroutine
from threading import Event
import time
from concert.connections.datatransfers import multicast, generate_sinograms
import numpy as np


def producer(consumer):
    for i in range(5):
        consumer.send(i)


class TestDataTransfers(ConcertTest):

    def setUp(self):
        super(TestDataTransfers, self).setUp()
        self.data = None
        self.data_2 = None

    @coroutine
    def consume(self):
        while True:
            self.data = yield

    @coroutine
    def consume_2(self):
        while True:
            self.data_2 = yield

    def test_multicast(self):
        producer(multicast(self.consume(), self.consume_2()))
        self.assertEqual(self.data, 4)
        self.assertEqual(self.data_2, 4)

    def test_sinogenerator(self):
        n = 4
        ground_truth = np.zeros((n, n, n))

        def image_producer(consumer):
            for i in range(n):
                ground_truth[i] = np.arange(n ** 2).reshape(n, n) * (i + 1)
                consumer.send(ground_truth[i])

        sinograms = np.zeros((n, n, n))
        image_producer(generate_sinograms(sinograms))

        np.testing.assert_almost_equal(sinograms,
                                       ground_truth.transpose(1, 0, 2))
