"""
Optimization algorithms, i.e. the executive code which finds the
actual optimum.
"""

import logbook
from concert.base import LimitError


logger = logbook.Logger(__name__)


def halve(param, feedback, cmp_set, step, epsilon, max_iterations=100):
    """
    Simple optimizer based on interval halving. Optimize function y = f(x),
    where x is obtained from the value of parameter *param*, y is the
    result of *feedback*, *cmp_set* is a function for comparing the old
    and new values (it also swaps old value for the new one), *epsilon*
    is the precision to which we want to optimize and *max_iterations*
    limits the number of iterations.
    """
    direction = 1
    i = 0
    data = []
    value = feedback()
    data.append((param.get().result(), value))

    def turn(direction, step):
        return -direction, step / 2.0

    while i < max_iterations:
        try:
            param.set(param.get().result() + direction * step).wait()
            value = feedback()
            point_reached = step < epsilon

            if point_reached:
                break

            if not cmp_set(value):
                direction, step = turn(direction, step)

            data.append((param.get().result(), value))
            logger.debug("value: %g, parameter value: %s" %
                        (value, str(param.get().result())))
        except LimitError:
            direction, step = turn(direction, step)
        i += 1

    return data
