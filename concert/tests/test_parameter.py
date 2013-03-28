import unittest
import logbook
import quantities as q
from concert.base import *
from testfixtures import ShouldRaise, compare


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_read_only_parameter(self):
        def getter():
            return 0

        parameter = Parameter('foo', fget=getter)
        self.assertTrue(parameter.is_readable())
        self.assertFalse(parameter.is_writable())

        compare(parameter.get(), 0)

        with ShouldRaise(WriteAccessError('foo')):
            parameter.set(None)

    def test_write_only_parameter(self):
        def setter(value):
            pass

        parameter = Parameter('foo', fset=setter)
        self.assertTrue(parameter.is_writable())
        self.assertFalse(parameter.is_readable())

        parameter.set(None)

        with ShouldRaise(ReadAccessError('foo')):
            parameter.get()

    def test_invalid_unit(self):
        def setter(value):
            pass

        parameter = Parameter('foo', fset=setter, unit=q.mm)
        parameter.set(2 * q.mm)

        with ShouldRaise(UnitError):
            parameter.set(2 * q.s)

    def test_limiter(self):
        def setter(value):
            pass

        def limit(value):
            return value >= 0 and value <= 1

        parameter = Parameter('foo', fset=setter, limiter=limit)
        parameter.set(0)
        parameter.set(0.5)
        parameter.set(1)

        with ShouldRaise(LimitError):
            parameter.set(1.5)

    def test_callback(self):
        args = []

        class Collector(object):
            def setter(self, value):
                self.value = value

            def getter(self):
                return self.value

        def callback(parameter):
            args.append(parameter.get())

        collector = Collector()
        parameter = Parameter('foo',
                              fget=collector.getter,
                              fset=collector.setter)
        parameter.subscribe(callback)

        parameter.set(0.0)
        parameter.set(1.0)
        parameter.set(2.3)

        compare(args, [0.0, 1.0, 2.3])
