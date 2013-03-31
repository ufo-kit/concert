import unittest
import logbook
import quantities as q
from concert.base import *
from testfixtures import ShouldRaise, compare


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def test_names(self):
        with ShouldRaise(ValueError):
            Parameter('1pm')

        with ShouldRaise(ValueError):
            Parameter('current position')

        Parameter('this-is-correct')
        Parameter('this_too')

    def test_read_only_parameter(self):
        def getter():
            return 0

        parameter = Parameter('foo', fget=getter)
        self.assertTrue(parameter.is_readable())
        self.assertFalse(parameter.is_writable())

        compare(parameter.get().result(), 0)

        with ShouldRaise(WriteAccessError('foo')):
            parameter.set(None).result()

    def test_write_only_parameter(self):
        def setter(value):
            pass

        parameter = Parameter('foo', fset=setter)
        self.assertTrue(parameter.is_writable())
        self.assertFalse(parameter.is_readable())

        parameter.set(None).result()

        with ShouldRaise(ReadAccessError('foo')):
            parameter.get().result()

    def test_invalid_unit(self):
        def setter(value):
            pass

        parameter = Parameter('foo', fset=setter, unit=q.mm)
        parameter.set(2 * q.mm).result()

        with ShouldRaise(UnitError):
            parameter.set(2 * q.s).result()

    def test_limiter(self):
        def setter(value):
            pass

        def limit(value):
            return value >= 0 and value <= 1

        parameter = Parameter('foo', fset=setter, limiter=limit)
        parameter.set(0).result()
        parameter.set(0.5).result()
        parameter.set(1).result()

        with ShouldRaise(LimitError):
            parameter.set(1.5).result()
