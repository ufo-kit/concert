from concert.base import Parameter, ParameterError
from concert.devices.base import Device
from concert.session.utils import get_default_table
from concert.tests import TestCase


class MockDevice(Device):

    readonly = Parameter()
    writeonly = Parameter()

    def __init__(self):
        super(MockDevice, self).__init__()

    def _get_readonly(self):
        return 1

    def _set_writeonly(self, value):
        pass


class TestDevice(TestCase):

    def setUp(self):
        super(TestDevice, self).setUp()
        self.device = MockDevice()

    def test_accessor_functions(self):
        self.assertEqual(self.device.get_readonly().result(), 1)
        self.device.set_writeonly(0).join()

    def test_iterable(self):
        for param in self.device:
            self.assertTrue(param.name in ('readonly', 'writeonly'))

    def test_invalid_parameter(self):
        def query_invalid_param():
            self.device['foo']

        self.assertRaises(ParameterError, query_invalid_param)

    # def test_str(self):
    #     table = get_default_table(["Parameter", "Value"])
    #     table.border = False
    #     table.add_row(["readonly", "1"])
    #     table.add_row(["state", Device.NA])
    #     self.assertEqual(str(self.device), table.get_string())

    def test_context_manager(self):
        # This is just a functional test, we don't cover synchronization issues
        # here.
        with self.device as d:
            v = d.readonly
            d.writeonly = 2

    def test_parameter_lock_acquisition(self):
        with self.device['writeonly']:
            pass
