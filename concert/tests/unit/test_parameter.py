from concert.quantities import q
from concert.tests import TestCase
from concert.base import (Parameterizable, Parameter, Quantity, State, SoftLimitError,
                          transition, LockError)


class BaseDevice(Parameterizable):

    def __init__(self):
        super(BaseDevice, self).__init__()


class FooDevice(BaseDevice):

    state = State(default='standby')

    foo = Quantity(q.m, transition=transition(source='*', target='moved'))

    def __init__(self, default):
        super(FooDevice, self).__init__()
        self._value = default

    def _get_foo(self):
        return self._value

    def _set_foo(self, value):
        self._value = value

    param = Parameter(fget=_get_foo, fset=_set_foo)


class RestrictedFooDevice(FooDevice):
    def __init__(self, lower, upper):
        super(RestrictedFooDevice, self).__init__(0 * q.mm)
        self['foo'].lower = lower
        self['foo'].upper = upper


class TestParameterizable(TestCase):

    def setUp(self):
        super(TestParameterizable, self).setUp()
        self.foo1 = FooDevice(42 * q.m)
        self.foo2 = FooDevice(23 * q.m)

    def test_param(self):
        self.foo1.param = 15
        self.assertEqual(self.foo1.param, 15)

    def test_property_identity(self):
        self.foo1.foo = 15 * q.m
        self.assertEqual(self.foo1.foo, 15 * q.m)
        self.assertEqual(self.foo2.foo, 23 * q.m)

    def test_func_identity(self):
        self.foo1.set_foo(15 * q.m).join()
        self.assertEqual(self.foo1.get_foo().result(), 15 * q.m)
        self.assertEqual(self.foo2.get_foo().result(), 23 * q.m)

    def test_parameter_property(self):
        self.assertEqual(self.foo1['foo'].unit, q.m)

    def test_saving(self):
        m = FooDevice(0 * q.mm)

        m.foo = 1 * q.mm
        m.stash().join()
        m.foo = 2 * q.mm
        m.stash().join()
        m.foo = 0.123 * q.mm
        m.foo = 1.234 * q.mm

        m.restore().join()
        self.assertEqual(m.foo, 2 * q.mm)

        m.restore().join()
        self.assertEqual(m.foo, 1 * q.mm)

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

    def test_state(self):
        self.assertEqual(self.foo1.state, 'standby')
        self.foo1.foo = 2 * q.m
        self.assertEqual(self.foo1.state, 'moved')

    def test_lock(self):
        self.foo1['foo'].lock()
        self.assertTrue(self.foo1['foo'].locked)
        with self.assertRaises(LockError):
            self.foo1.foo = 1 * q.m

        # This must pass
        self.foo1.param = 1 * q.m

        # Unlock and test if writable
        self.foo1['foo'].unlock()
        self.assertFalse(self.foo1['foo'].locked)
        self.foo1.foo = 1 * q.m

        # Lock the whole device
        self.foo1.lock()
        with self.assertRaises(LockError):
            self.foo1.param = 1 * q.m
        with self.assertRaises(LockError):
            self.foo1.foo = 1 * q.m

        # Unlock the whole device and test if writable
        self.foo1.unlock()
        self.foo1.param = 1 * q.m
        self.foo1.foo = 1 * q.m

    def test_permanent_lock(self):
        self.foo1['foo'].lock(permanent=True)
        self.assertTrue(self.foo1['foo'].locked)
        with self.assertRaises(LockError):
            self.foo1.foo = 1 * q.m

        with self.assertRaises(LockError):
            self.foo1['foo'].unlock()

        # While device
        self.foo2.lock(permanent=True)
        with self.assertRaises(LockError):
            self.foo2.unlock()
