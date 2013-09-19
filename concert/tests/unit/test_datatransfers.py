import time
import numpy as np
from threading import Event
from concert.quantities import q
from concert.tests.base import suppressed_logging, ConcertTest
from concert.devices.motors.dummy import Motor
from concert.processes.base import coroutine, multicast, inject
from concert.processes.sinks import generate_sinograms
from concert.processes.scan import Scanner


def producer(consumer):
    for i in range(5):
        consumer.send(i)


def generator():
    for i in range(5):
        yield i


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

    def test_injection(self):
        inject(generator(), self.consume())
        self.assertEqual(self.data, 4)

    def test_scan_generator(self):
        motor = Motor()
        scanner = Scanner(motor['position'], lambda: motor.position)
        scanner.minimum = 0 * q.mm
        scanner.maximum = 2 * q.cm
        scanner.gen(self.consume())
        self.assertEqual(self.data[0], 2 * q.cm)
        self.assertEqual(self.data[1], 2 * q.cm)
