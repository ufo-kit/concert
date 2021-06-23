from concert.base import Parameter, ParameterError
from concert.devices.base import abort_all_devices, Device
from concert.tests import TestCase, suppressed_logging
from concert.devices.scales.dummy import Scales, TarableScales
from concert.devices.pumps.dummy import Pump
from concert.devices.io.dummy import IO
from concert.devices.monochromators.dummy import Monochromator
from concert.devices.shutters.dummy import Shutter
from concert.devices.storagerings.dummy import StorageRing
from concert.devices.photodiodes.dummy import PhotoDiode
from concert.devices.lightsources.dummy import LightSource


@suppressed_logging
def test_dummies():
    Scales()
    TarableScales()
    Pump()
    IO()
    Monochromator()
    Shutter()
    StorageRing()
    PhotoDiode()
    LightSource()


class MockDevice(Device):

    readonly = Parameter()
    writeonly = Parameter()

    def __init__(self):
        super(MockDevice, self).__init__()
        self.aborted = False

    async def _get_readonly(self):
        return 1

    async def _set_writeonly(self, value):
        pass

    async def _abort(self):
        self.aborted = True


class TestDevice(TestCase):

    def setUp(self):
        super(TestDevice, self).setUp()
        self.device = MockDevice()

    async def test_accessor_functions(self):
        self.assertEqual(await self.device.get_readonly(), 1)
        await self.device.set_writeonly(0)

    def test_iterable(self):
        for param in self.device:
            self.assertTrue(param.name in ('readonly', 'writeonly'))

    def test_invalid_parameter(self):
        def query_invalid_param():
            self.device['foo']

        self.assertRaises(ParameterError, query_invalid_param)

    def test_str(self):
        from concert.session.utils import get_default_table
        table = get_default_table(["Parameter", "Value"])
        table.border = False
        table.add_row(["readonly", "1"])
        table.add_row(["writeonly", "N/A"])
        self.assertEqual(str(self.device), table.get_string())

    def test_context_manager(self):
        # This is just a functional test, we don't cover synchronization issues
        # here.
        with self.device as d:
            d.readonly
            d.writeonly = 2

    def test_parameter_lock_acquisition(self):
        with self.device['writeonly']:
            pass

    async def test_abort(self):
        await self.device.abort()
        self.assertTrue(self.device.aborted)

    async def test_abort_all(self):
        devices = [self.device, MockDevice()]
        await abort_all_devices(devices)

        for device in devices:
            self.assertTrue(device.aborted)
