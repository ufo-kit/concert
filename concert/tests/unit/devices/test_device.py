from concert.base import Parameter, ParameterError
from concert.devices.base import Device
from concert.tests import TestCase
from concert.devices.scales.dummy import Scales, TarableScales
from concert.devices.pumps.dummy import Pump
from concert.devices.io.dummy import IO
from concert.devices.monochromators.dummy import Monochromator
from concert.devices.shutters.dummy import Shutter
from concert.devices.storagerings.dummy import StorageRing
from concert.devices.photodiodes.dummy import PhotoDiode
from concert.devices.lightsources.dummy import LightSource


class MockDevice(Device):

    readonly = Parameter()
    writeonly = Parameter()

    async def __ainit__(self):
        await super(MockDevice, self).__ainit__()
        self.aborted = False

    async def _get_readonly(self):
        return 1

    async def _set_writeonly(self, value):
        pass

    async def _emergency_stop(self):
        self.aborted = True


class TestDevice(TestCase):

    async def asyncSetUp(self):
        await super(TestDevice, self).asyncSetUp()
        self.device = await MockDevice()

    async def test_dummies(self):
        await Scales()
        await TarableScales()
        await Pump()
        await IO()
        await Monochromator()
        await Shutter()
        await StorageRing()
        await PhotoDiode()
        await LightSource()

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

    async def test_context_manager(self):
        # This is just a functional test, we don't cover synchronization issues
        # here.
        async with self.device as d:
            await d.get_readonly()
            await d.set_writeonly(2)

    async def test_parameter_lock_acquisition(self):
        async with self.device['writeonly']:
            pass

    async def test_emergency_stop(self):
        await self.device.emergency_stop()
        self.assertTrue(self.device.aborted)
