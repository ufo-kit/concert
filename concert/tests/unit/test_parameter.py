from concert.quantities import q
from concert.base import *
from concert.tests.base import ConcertTest


class BaseDevice(Parameterizable):

    def __init__(self):
        param = Parameter('foo', fget=self._get, fset=self._set)
        super(BaseDevice, self).__init__([param])

    def _get(self):
        raise NotImplementedError

    def _set(self, value):
        raise NotImplementedError


class FooDevice(BaseDevice):

    def __init__(self, proxy):
        super(FooDevice, self).__init__()
        self.proxy = proxy

    def _get(self):
        return self.proxy.get()

    def _set(self, value):
        self.proxy.set(value)


class Proxy(object):

    def __init__(self, default):
        self.value = default

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def empty_setter(v):
    pass


class TestParameterizable(ConcertTest):

    def setUp(self):
        super(TestParameterizable, self).setUp()
        proxy1 = Proxy(42)
        proxy2 = Proxy(23)
        self.foo1 = FooDevice(proxy1)
        self.foo2 = FooDevice(proxy2)

    def test_property_identity(self):
        self.foo1.foo = 15
        self.assertEqual(self.foo1.foo, 15)
        self.assertEqual(self.foo2.foo, 23)

    def test_func_identity(self):
        self.foo1.set_foo(15).wait()
        self.assertEqual(self.foo1.get_foo().result(), 15)
        self.assertEqual(self.foo2.get_foo().result(), 23)


class TestParameter(ConcertTest):

    def test_names(self):
        self.assertRaises(ValueError, Parameter, '1pm')
        self.assertRaises(ValueError, Parameter, 'current position')

        Parameter('this-is-correct')
        Parameter('this_too')

    def test_read_only_parameter(self):
        def getter():
            return 0

        parameter = Parameter('foo', fget=getter)
        self.assertTrue(parameter.is_readable())
        self.assertFalse(parameter.is_writable())

        self.assertEqual(parameter.get().result(), 0)

        with self.assertRaises(WriteAccessError) as ctx:
            parameter.set(None).result()

        self.assertEqual("parameter `foo' cannot be written",
                         ctx.exception.message)

    def test_write_only_parameter(self):
        parameter = Parameter('foo', fset=empty_setter)
        self.assertTrue(parameter.is_writable())
        self.assertFalse(parameter.is_readable())

        parameter.set(1).result()

        with self.assertRaises(ReadAccessError) as ctx:
            parameter.get().result()

        self.assertEqual("parameter `foo' cannot be read",
                         ctx.exception.message)

    def test_invalid_unit(self):
        parameter = Parameter('foo', fset=empty_setter, unit=q.mm)
        parameter.set(2 * q.mm).result()

        self.assertRaises(UnitError, parameter.set(2 * q.s).result)

    def test_soft_limit(self):
        parameter = Parameter('foo', fset=empty_setter, unit=q.mm,
                              lower=2 * q.mm, upper=4 * q.mm)

        parameter.set(2.3 * q.mm).wait()
        parameter.set(2 * q.mm).wait()
        parameter.set(4 * q.mm).wait()

        self.assertRaises(SoftLimitError, parameter.set(4.2 * q.mm).wait)

    def test_hard_limit(self):
        def setter(value):
            pass

        class Limited(object):
            in_limit = False

            def __call__(self):
                return self.in_limit

        in_limit = Limited()

        parameter = Parameter('foo', fset=setter, in_hard_limit=in_limit)
        parameter.set(0).result()
        parameter.set(0.5).result()
        parameter.set(1).result()

        with self.assertRaises(HardLimitError):
            in_limit.in_limit = True
            parameter.set(1.5).result()
