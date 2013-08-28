import unittest
import logbook
from concert.devices.io.dummy import IO
from concert.base import ReadAccessError, WriteAccessError


class TestDummyIO(unittest.TestCase):

    def setUp(self):
        self.io_device = IO()
        self.handler = logbook.NullHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_read_port(self):
        self.assertEqual(self.io_device.exposure, 0)
        self.assertEqual(self.io_device.busy, 1)

    def test_write_port(self):
        self.io_device.acq_enable = 1
        self.assertEqual(self.io_device._ports
                         [self.io_device["acq_enable"].port_id], 1)

    def test_read_not_readable_port(self):
        self.assertRaises(ReadAccessError,
                          self.io_device["acq_enable"].get().wait)

    def test_write_not_writable_port(self):
        self.assertRaises(WriteAccessError,
                          self.io_device["busy"].set(1).wait)
