from concert.devices.io.dummy import IO
from concert.base import ReadAccessError, WriteAccessError
from concert.tests import TestCase


class TestDummyIO(TestCase):

    def setUp(self):
        super(TestDummyIO, self).setUp()
        self.io_device = IO()

    def test_read_port(self):
        self.assertEqual(self.io_device.exposure, 0)
        self.assertEqual(self.io_device.busy, 1)

    def test_write_port(self):
        self.io_device.acq_enable = 1

    def test_read_not_readable_port(self):
        with self.assertRaises(ReadAccessError):
            self.io_device["acq_enable"].get().join()

    def test_write_not_writable_port(self):
        with self.assertRaises(WriteAccessError):
            self.io_device["busy"].set(1).join()
