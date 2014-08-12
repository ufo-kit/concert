import time
from concert.quantities import q
from concert.tests import TestCase
from concert.base import (Parameterizable, Parameter, Quantity, State, SoftLimitError,
                          transition, check, LockError, ParameterError,
                          UnitError, WriteAccessError)
from concert.helpers import WaitError


class BaseDevice(Parameterizable):

    def __init__(self):
        super(BaseDevice, self).__init__()


class FooDevice(BaseDevice):

    state = State(default='standby')

    no_write = Parameter()
    foo = Quantity(q.m, check=check(source='*', target='moved'))

    def __init__(self, default):
        super(FooDevice, self).__init__()
        self._value = default

    def _get_foo(self):
        return self._value

    @transition(target='moved')
    def _set_foo(self, value):
        self._value = value

    param = Parameter(fget=_get_foo, fset=_set_foo)


class RestrictedFooDevice(FooDevice):
    def __init__(self, lower, upper):
        super(RestrictedFooDevice, self).__init__(0 * q.mm)
        self['foo'].lower = lower
        self['foo'].upper = upper


class AccessorCheckDevice(Parameterizable):

    foo = Quantity(q.m)

    def __init__(self, future, check):
        super(AccessorCheckDevice, self).__init__()
        self.check = check
        self.future = future
        self._value = 0 * q.mm

    def _set_foo(self, value):
        self.check(self.future)
        self._value = value
        time.sleep(0.01)

    def _get_foo(self):
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

    def test_func_identity(self):
        self.foo1.set_foo(15 * q.m).join()
        self.assertEqual(self.foo1.get_foo().result(), 15 * q.m)
        self.assertEqual(self.foo2.get_foo().result(), 23 * q.m)

    def test_wait(self):
        self.foo1.foo = 1 * q.m
        self.foo1['foo'].wait(1 * q.m, eps=1e-3 * q.m)

        # No change on the parameter, timeout must take place
        with self.assertRaises(WaitError):
            self.foo1['foo'].wait(0 * q.m, timeout=1e-5 * q.s)


class TestParameterizable(TestCase):

    def setUp(self):
        super(TestParameterizable, self).setUp()
        self.device = FooDevice(0 * q.mm)

    def test_param(self):
        self.device.param = 15
        self.assertEqual(self.device.param, 15)

    def test_is_writable(self):
        self.assertTrue(self.device['foo'].writable)
        self.assertFalse(self.device['no_write'].writable)

        with self.assertRaises(WriteAccessError):
            self.device.no_write = 42


    def test_saving(self):
        self.device.foo = 1 * q.mm
        self.device.stash().join()
        self.device.foo = 2 * q.mm
        self.device.stash().join()
        self.device.foo = 0.123 * q.mm
        self.device.foo = 1.234 * q.mm

        self.device.restore().join()
        self.assertEqual(self.device.foo, 2 * q.mm)

        self.device.restore().join()
        self.assertEqual(self.device.foo, 1 * q.mm)

    def test_state(self):
        self.assertEqual(self.device.state, 'standby')
        self.device.foo = 2 * q.m
        self.assertEqual(self.device.state, 'moved')

    def test_manual_lock(self):
        self.device['foo'].lock()
        self.assertTrue(self.device['foo'].locked)

        with self.assertRaises(LockError):
            self.device.foo = 1 * q.m

        # This must pass
        self.device.param = 1 * q.m

        # Unlock and test if writable
        self.device['foo'].unlock()
        self.assertFalse(self.device['foo'].locked)
        self.device.foo = 1 * q.m

        # Lock the whole device
        self.device.lock()

        with self.assertRaises(LockError):
            self.device.param = 1 * q.m

        with self.assertRaises(LockError):
            self.device.foo = 1 * q.m

        # Unlock the whole device and test if writable
        self.device.unlock()
        self.device.param = 1 * q.m
        self.device.foo = 1 * q.m

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

    def test_wait_on_future(self):
        def check(future):
            if future:
                self.assertTrue(future.done())

        d1 = AccessorCheckDevice(None, check)
        f1 = d1.set_foo(2 * q.mm)
        d2 = AccessorCheckDevice(f1, check)
        f2 = d2.set_foo(5 * q.mm, wait_on=f1)
        d3 = AccessorCheckDevice(f2, check)
        d3.get_foo(wait_on=f2).join()



class TestParameter(TestCase):

    def test_saving(self):
        device = FooDevice(0 * q.mm)
        device['foo'].stash().join()
        device.foo = 1 * q.mm
        device['foo'].restore().join()
        self.assertEqual(device.foo, 0 * q.mm)

        with self.assertRaises(ParameterError):
            device['no_write'].stash().join()

        with self.assertRaises(ParameterError):
            device['no_write'].restore().join()


class TestQuantity(TestCase):

    def test_soft_limit_change(self):
        limited = FooDevice(0 * q.mm)
        limited['foo'].lower = -2 * q.mm
        limited['foo'].upper = 2 * q.mm

        limited.foo = -1.5 * q.mm
        limited.foo = +1.5 * q.mm

        with self.assertRaises(SoftLimitError):
            limited.foo = 2.5 * q.mm

    def test_soft_limit_restriction(self):
        limited = RestrictedFooDevice(-2 * q.mm, 2 * q.mm)
        self.assertEqual(limited['foo'].lower, -2 * q.mm)
        self.assertEqual(limited['foo'].upper, +2 * q.mm)

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
