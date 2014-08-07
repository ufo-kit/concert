from concert.tests import TestCase
from concert.base import TransitionNotAllowed
from concert.devices.io.base import IODeviceError
from concert.devices.io.dummy import IO, Signal


class TestIO(TestCase):

    def setUp(self):
        self.io = IO(port_value=0)
        self.port = 0

    def test_read(self):
        self.assertEquals(0, self.io.read_port(self.port))

    def test_write(self):
        value = 1
        self.io.write_port(self.port, value)
        self.assertEquals(value, self.io.read_port(self.port))

    def test_non_existent_read(self):
        self.assertRaises(IODeviceError, self.io.read_port, 1)

    def test_non_existent_write(self):
        self.assertRaises(IODeviceError, self.io.write_port, 1, 0)


class TestSignal(TestCase):

    def setUp(self):
        self.signal = Signal()

    def test_on(self):
        self.signal.on()
        self.assertEqual(self.signal.state, 'on')

        with self.assertRaises(TransitionNotAllowed):
            self.signal.on()

    def test_off(self):
        self.signal.on()
        self.signal.off()
        self.assertEqual(self.signal.state, 'off')

        with self.assertRaises(TransitionNotAllowed):
            self.signal.off()
