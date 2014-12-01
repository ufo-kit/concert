import time
from concert.quantities import q
from concert.tests import suppressed_logging
from concert.helpers import measure


@suppressed_logging
def test_measure_execution():
    @measure(return_result=True)
    def sleeping():
        time.sleep(0.001)
        return 123

    result, elapsed = sleeping()
    assert(result == 123)
    assert(elapsed > 0.001 * q.s)
    assert(elapsed < 0.010 * q.s)
