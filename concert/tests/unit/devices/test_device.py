import unittest
import logbook
from testfixtures import ShouldRaise, compare
from concert.base import Parameter, ParameterError
from concert.devices.base import Device
from concert.ui import get_default_table


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
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_accessor_functions(self):
        compare(self.device.get_readonly().result(), 1)
        self.device.set_writeonly(None).wait()

    def test_iterable(self):
        for param in self.device:
            self.assertTrue(param.name in ('readonly', 'writeonly', 'state'))

    def test_get_parameter(self):
        compare(self.device['readonly'], self.device.params[0])
        compare(self.device['writeonly'], self.device.params[1])

    def test_invalid_paramter(self):
        with ShouldRaise(ParameterError):
            param = self.device['foo']
            compare(param, None)

    def test_str(self):
        table = get_default_table(["Parameter", "Value"])
        table.border = False
        table.add_row(["readonly", "1"])
        table.add_row(["state", Device.NA])

        compare(str(self.device), table.get_string())

    def test_context_manager(self):
        # This is just a functional test, we don't cover synchronization issues
        # here.
        with self.device as d:
            v = d.readonly
            d.writeonly = 2
