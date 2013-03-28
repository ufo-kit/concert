import unittest
import logbook
import time
from testfixtures import ShouldRaise, compare
from concert.base import Device, Parameter, ParameterError


class MockDevice(Device):
    def __init__(self):
        def setter(value):
            pass

        def getter():
            return 1

        self.params = [Parameter('readonly', fget=getter),
                       Parameter('writeonly', fset=setter)]

        super(MockDevice, self).__init__(self.params)


class TestDevice(unittest.TestCase):
    def setUp(self):
        self.device = MockDevice()
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_iterable(self):
        for param in self.device:
            self.assertTrue(param.name in ('readonly', 'writeonly'))

    def test_get_parameter(self):
        compare(self.device['readonly'], self.device.params[0])
        compare(self.device['writeonly'], self.device.params[1])

    def test_invalid_paramter(self):
        with ShouldRaise(ParameterError):
            param = self.device['foo']

    def test_str(self):
        compare(str(self.device), "readonly = 1")
