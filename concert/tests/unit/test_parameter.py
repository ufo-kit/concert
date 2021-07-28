import asyncio
import time
import numpy as np
from concert.coroutines.base import start, WaitError
from concert.quantities import q
from concert.tests import TestCase
from concert.base import (Parameterizable, Parameter, Quantity, State, transition, check,
                          SelectionError, SoftLimitError, LockError, ParameterError, UnitError,
                          WriteAccessError)
from concert.devices.dummy import SelectionDevice


class BaseDevice(Parameterizable):

    def __init__(self):
        super(BaseDevice, self).__init__()


async def _test_setter(device, value):
    device._test_value = value


async def _test_getter(device):
    return device._test_value


async def _test_target(device):
    return 10 * q.mm


class FooDevice(BaseDevice):

    state = State(default='standby')

    no_write = Parameter()
    foo = Quantity(q.m, check=check(source='*', target='moved'))
    bar = Quantity(q.m)
    test = Quantity(q.m, fset=_test_setter, fget=_test_getter)

    def __init__(self, default):
        super(FooDevice, self).__init__()
        self._value = default
        self._param_value = 0 * q.mm
        self._test_value = 0 * q.mm

    async def _get_foo(self):
        return self._value

    @transition(target='moved')
    async def _set_foo(self, value):
        self._value = value

    async def _get_bar(self):
        return 5 * q.m

    param = Parameter(fget=_get_foo, fset=_set_foo)

    async def _get_param(self):
        return self._param_value

    async def _set_param(self, value):
        self._param_value = value

    param = Parameter(fget=_get_param, fset=_set_param)


class FooDeviceTargetValue(BaseDevice):
    foo = Quantity(q.mm)

    def __init__(self, value):
        super(BaseDevice, self).__init__()
        self._value = value
        self._test_value = value

    async def _get_foo(self):
        delta = 1 * q.mm
        return self._value + delta

    async def _set_foo(self, value):
        self._value = value

    async def _get_target_foo(self):
        return self._value

    def test_setter(self, value):
        self._test_value = value

    def test_getter(self):
        return self._test_value

    def test_target(self):
        return 10 * q.mm

    test = Quantity(q.mm, fset=_test_setter, fget=_test_getter, fget_target=_test_target)


class FooDeviceTagetValue(BaseDevice):
    foo = Quantity(q.mm)
    test = Quantity(q.mm, fset=_test_setter, fget=_test_getter, fget_target=_test_target)

    def __init__(self, value):
        super(BaseDevice, self).__init__()
        self._value = value

    def _get_foo(self):
        delta = np.random.randint(low=1, high=100) * q.mm
        return self._value + delta

    def _set_foo(self, value):
        self._value = value

    def _get_target_foo(self):
        return self._value


class RestrictedFooDevice(FooDevice):
    def __init__(self, lower, upper):
        super(RestrictedFooDevice, self).__init__(0 * q.mm)
        self['foo'].lower = lower
        self['foo'].upper = upper


class ExternalLimitDevice(BaseDevice):

    foo = Quantity(q.mm,
                   external_lower_getter=lambda: -5 * q.mm,
                   external_upper_getter=lambda: 5 * q.mm)

    def __init__(self, value):
        super(ExternalLimitDevice, self).__init__()
        self._value = value


class AccessorCheckDevice(Parameterizable):

    foo = Quantity(q.m)

    def __init__(self, future, check):
        super(AccessorCheckDevice, self).__init__()
        self.check = check
        self.future = future
        self._value = 0 * q.mm

    async def _set_foo(self, value):
        self.check(self.future)
        self._value = value
        time.sleep(0.01)

    async def _get_foo(self):
        self.check(self.future)
        time.sleep(0.01)
        return self._value


class TestDescriptor(TestCase):

    def setUp(self):
        super(TestDescriptor, self).setUp()
        self.foo1 = FooDevice(42 * q.m)
        self.foo2 = FooDevice(23 * q.m)

    def test_property_identity(self):
        self.foo1.foo = 15 * q.m
        self.assertEqual(self.foo1.foo, 15 * q.m)
        self.assertEqual(self.foo2.foo, 23 * q.m)

    async def test_func_identity(self):
        await self.foo1.set_foo(15 * q.m)
        self.assertEqual(await self.foo1.get_foo(), 15 * q.m)
        self.assertEqual(await self.foo2.get_foo(), 23 * q.m)

    async def test_wait(self):
        await self.foo1.set_foo(1 * q.m)
        await self.foo1['foo'].wait(1 * q.m, eps=1e-3 * q.m)

        # No change on the parameter, timeout must take place
        with self.assertRaises(WaitError):
            await self.foo1['foo'].wait(0 * q.m, timeout=1e-5 * q.s)


class TestParameterizable(TestCase):

    def setUp(self):
        super(TestParameterizable, self).setUp()
        self.device = FooDevice(0 * q.mm)

    async def test_param(self):
        await self.device.set_param(15)
        self.assertEqual(await self.device.get_param(), 15)

    def test_is_writable(self):
        self.assertTrue(self.device['foo'].writable)
        self.assertFalse(self.device['no_write'].writable)
        self.assertTrue(self.device['param'].writable)

        with self.assertRaises(WriteAccessError):
            self.device.no_write = 42

    async def test_saving(self):
        await self.device.set_foo(1 * q.mm)
        await self.device.stash()
        await self.device.set_foo(2 * q.mm)
        await self.device.stash()
        await self.device.set_foo(0.123 * q.mm)
        await self.device.set_foo(1.234 * q.mm)

        await self.device.restore()
        self.assertEqual(await self.device.get_foo(), 2 * q.mm)

        await self.device.restore()
        self.assertEqual(await self.device.get_foo(), 1 * q.mm)

    async def test_manual_lock(self):
        self.device['foo'].lock()
        self.assertTrue(self.device['foo'].locked)

        with self.assertRaises(LockError):
            await self.device.set_foo(1 * q.m)

        # This must pass
        await self.device.set_param(1 * q.m)

        # Unlock and test if writable
        self.device['foo'].unlock()
        self.assertFalse(self.device['foo'].locked)
        await self.device.set_foo(1 * q.m)

        # Lock the whole device
        self.device.lock()

        with self.assertRaises(LockError):
            await self.device.set_param(1 * q.m)

        with self.assertRaises(LockError):
            await self.device.set_foo(1 * q.m)

        # Unlock the whole device and test if writable
        self.device.unlock()
        await self.device.set_param(1 * q.m)
        await self.device.set_foo(1 * q.m)

    def test_permanent_lock(self):
        self.device['foo'].lock(permanent=True)
        self.assertTrue(self.device['foo'].locked)

        with self.assertRaises(LockError):
            self.device.foo = 1 * q.m

        with self.assertRaises(LockError):
            self.device['foo'].unlock()

    def test_permanent_device_lock(self):
        self.device.lock(permanent=True)

        with self.assertRaises(LockError):
            self.device.unlock()

    async def test_wait_on_future(self):
        def check(future):
            if future:
                self.assertTrue(future.done())

        d1 = AccessorCheckDevice(None, check)
        f1 = start(d1.set_foo(2 * q.mm))
        d2 = AccessorCheckDevice(f1, check)
        f2 = start(d2.set_foo(5 * q.mm, wait_on=f1))
        d3 = AccessorCheckDevice(f2, check)
        f3 = start(d3.get_foo(wait_on=f2))
        await asyncio.gather(f1, f2, f3)


class TestParameter(TestCase):

    async def test_saving(self):
        device = FooDevice(0 * q.mm)
        await device['foo'].stash()
        await device.set_foo(1 * q.mm)
        await device['foo'].restore()
        self.assertEqual(await device.get_foo(), 0 * q.mm)

        with self.assertRaises(ParameterError):
            await device['no_write'].stash()

        with self.assertRaises(ParameterError):
            await device['no_write'].restore()

    async def test_saving_with_target_value(self):
        device = FooDeviceTargetValue(0 * q.mm)
        device['foo'].upper = None
        device['foo'].lower = None
        await device['foo'].stash()
        self.assertEqual(device['foo'].target_readable, True)
        await device.set_foo(1 * q.mm)
        self.assertEqual(device._value, 1 * q.mm)
        await device['foo'].restore()
        self.assertEqual(device._value, 0 * q.mm)

    def test_readonly_value(self):
        device = FooDevice(0 * q.mm)
        self.assertEqual(device.bar, 5 * q.m)
        self.assertEqual(device['bar'].writable, False)
        with self.assertRaises(WriteAccessError):
            device.bar = 1 * q.m

    def test_setter_getter_from_constructor(self):
        device = FooDevice(0 * q.mm)
        device.test = 1 * q.mm
        self.assertEqual(device['test'].writable, True)
        self.assertEqual(device.test, 1 * q.mm)

    def test_setter_getter_from_constructor_target(self):
        device = FooDeviceTargetValue(0 * q.mm)
        device.test = 1 * q.mm
        self.assertEqual(device.test, 1 * q.mm)
        self.assertEqual(device['test'].target_readable, True)
        self.assertEqual(device['test'].target, 10 * q.mm)

    def test_name_for_log(self):
        device = FooDevice(0 * q.mm)
        device.foo = 1 * q.mm
        self.assertEqual(device.name_for_log, 'device')


class TestQuantity(TestCase):

    def test_soft_limit_change(self):
        limited = FooDevice(0 * q.mm)
        limited['foo'].lower = -2 * q.mm
        limited['foo'].upper = 2 * q.mm

        limited.foo = -1.5 * q.mm
        limited.foo = +1.5 * q.mm

        with self.assertRaises(SoftLimitError):
            limited.foo = 2.5 * q.mm

    async def test_soft_limit_restriction(self):
        limited = RestrictedFooDevice(-2 * q.mm, 2 * q.mm)
        self.assertEqual(limited['foo'].lower, -2 * q.mm)
        self.assertEqual(limited['foo'].upper, +2 * q.mm)

    def test_setting_soft_limits_to_none(self):
        limited = RestrictedFooDevice(-2 * q.mm, 2 * q.mm)
        limited['foo'].upper = None
        limited.foo = 3 * q.mm
        limited['foo'].lower = None
        limited.foo = -3 * q.mm

    def test_parameter_property(self):
        device = FooDevice(42 * q.m)
        self.assertEqual(device['foo'].unit, q.m)

        with self.assertRaises(UnitError):
            device.foo = 2 * q.s

    def test_limits_lock(self):
        device = FooDevice(10 * q.mm)
        device['foo'].lock_limits()
        with self.assertRaises(LockError):
            device['foo'].lower = -10 * q.mm

        device['foo'].lock_limits(True)
        with self.assertRaises(LockError):
            device['foo'].unlock_limits()

    def test_limits_setting(self):
        dev = FooDevice(0 * q.mm)
        dev['foo'].lower = -1 * q.m
        dev['foo'].upper = 1 * q.um

        with self.assertRaises(UnitError):
            dev['foo'].lower = -1 * q.deg

        with self.assertRaises(UnitError):
            dev['foo'].upper = 1 * q.deg

    def test_limit_bounds(self):
        dev = FooDevice(0 * q.m)
        dev['foo'].lower = -1 * q.m
        dev['foo'].upper = 1 * q.m

        with self.assertRaises(ValueError):
            dev['foo'].lower = 2 * q.m

        with self.assertRaises(ValueError):
            dev['foo'].upper = -2 * q.m

    async def test_external_limits(self):
        dev = ExternalLimitDevice(0 * q.mm)
        self.assertEqual(dev['foo'].lower, -5 * q.mm)
        self.assertEqual(dev['foo'].upper, 5 * q.mm)

        dev['foo'].upper = 2 * q.mm
        self.assertEqual(dev['foo'].upper, 2 * q.mm)

        dev['foo'].upper = 10 * q.mm
        self.assertEqual(dev['foo'].upper, 5 * q.mm)

        dev['foo'].lower = -2 * q.mm
        self.assertEqual(dev['foo'].lower, -2 * q.mm)

        dev['foo'].lower = -10 * q.mm
        self.assertEqual(dev['foo'].lower, -5 * q.mm)


class TestSelection(TestCase):

    def setUp(self):
        self.device = SelectionDevice()

    def test_correct_access(self):
        for i in range(3):
            self.device.selection = i

    def test_wrong_access(self):
        with self.assertRaises(SelectionError):
            self.device.selection = 4

    def test_iterable_access(self):
        np.testing.assert_equal(self.device['selection'].values, list(range(3)))
