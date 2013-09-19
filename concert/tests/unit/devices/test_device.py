from concert.base import Parameter, ParameterError
from concert.devices.base import Device
from concert.ui import get_default_table
from concert.tests.base import ConcertTest


class MockDevice(Device):

    def __init__(self):
        def setter(value):
            pass

        def getter():
            return 1

        self.params = [Parameter('readonly', fget=getter),
                       Parameter('writeonly', fset=setter)]

        super(MockDevice, self).__init__(self.params)


class TestDevice(ConcertTest):

    def setUp(self):
        super(TestDevice, self).setUp()
        self.device = MockDevice()

    def test_accessor_functions(self):
        self.assertEqual(self.device.get_readonly().result(), 1)
        self.device.set_writeonly(0).wait()

    def test_iterable(self):
        for param in self.device:
            self.assertTrue(param.name in ('readonly', 'writeonly', 'state'))

    def test_get_parameter(self):
        self.assertEqual(self.device['readonly'], self.device.params[0])
        self.assertEqual(self.device['writeonly'], self.device.params[1])

    def test_invalid_paramter(self):
        def query_invalid_param():
            self.device['foo']

        self.assertRaises(ParameterError, query_invalid_param)

    def test_str(self):
        table = get_default_table(["Parameter", "Value"])
        table.border = False
        table.add_row(["readonly", "1"])
        table.add_row(["state", Device.NA])

        self.assertEqual(str(self.device), table.get_string())

    def test_context_manager(self):
        # This is just a functional test, we don't cover synchronization issues
        # here.
        with self.device as d:
            v = d.readonly
            d.writeonly = 2

    def test_parameter_locks_exist(self):
        for param_name in ('state', 'readonly', 'writeonly'):
            self.assertEqual(self.device._lock, self.device[param_name].lock)

    def test_parameter_lock_acquisition(self):
        with self.device['writeonly']:
            pass
