"""
Focus related processes
"""
import logbook
from concert.optimization.scalar import Maximizer
from concert.base import LimitError
from concert.asynchronous import async, dispatcher


log = logbook.Logger(__name__)


class Focuser(object):

    FOUND = 'focus-found'

    def __init__(self, axis, epsilon, gradient_feedback):
        self._axis = axis
        self._epsilon = epsilon
        self._gradient_feedback = gradient_feedback

    @async
    def focus(self, step):
        """
        focus(step)
        A simple focusing process using gradient as a measure. It maximizes
        it and if the difference between to consequently taken points is
        smaller than an epsilon it stops the process. The gradient function,
        which depends on the position is not monotonic, thus the following
        case might happen:

        g(x1) = g(x2), x1 != x2 and x1 != x2 != axis position for global
        maximum of the gradient.

        g(x) is the gradient function dependent on position x.

        *Step* is the inital step by which to move the axis. If *blocking* is
        True, the method does not block and returns immediately. After the
        position of the global maximum is found, an appropriate message
        is sent.
        """
        direction = 1
        hits = 0
        maximizer = Maximizer(self._epsilon)

        def turn(direction, step):
            return -direction, step / 2.0

        while True:
            try:
                self._axis.move(direction * step).wait()
                gradient = self._gradient_feedback()
                point_reached = step < self._epsilon
                limit_reached = self._axis.in_hard_limit()
                worse = not maximizer.is_better(gradient)

                if point_reached or limit_reached or worse:
                    direction, step = turn(direction, step)

                if point_reached:
                    # Make sure we found the global maximum and not only
                    # two equal gradients but out of focus. This can happen
                    # if we are out of focus and by motor movement we go
                    # to the other side of the image plane with equal
                    # gradient.
                    hits += 1
                else:
                    hits = 0

                if hits == 2:
                    break

                maximizer.value = gradient
                log.debug("Gradient: %g, axis position: %s" %
                          (gradient, str(self._axis.position)))
            except LimitError:
                direction, step = turn(direction, step)

        dispatcher.send(self, self.FOUND)
        log.info("Maximum gradient: %g found at position: %s" %
                 (gradient, str(self._axis.position)))
