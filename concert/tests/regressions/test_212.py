from concert.quantities import q
from concert.tests import suppressed_logging, assert_almost_equal
from concert.tests.unit.test_parameter import ConvertingDevice


class LazyConvertingDevice(ConvertingDevice):
    def _set_foo(self, value):
        self._value = '{}'.format(value.magnitude)

    def _get_foo(self):
        return float(self._value) * q.count


@suppressed_logging
def test_different_quantity():
    device = LazyConvertingDevice()
    device.foo = 1 * q.km
    assert_almost_equal(1 * q.km, device.foo)
