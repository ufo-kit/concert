"""
Focus related processes
"""
import logbook
from concert.optimization.scalar import Maximizer
from concert.base import ConcertObject, LimitError


log = logbook.Logger(__name__)


class FocuserMessage(object):
    FOCUS_FOUND = "focus_found"


def _focus(axis, feedback, step, epsilon):
    direction = 1
    hits = 0
    maximizer = Maximizer(epsilon)

    def turn(direction, step):
        return -direction, step / 2.0

    while True:
        try:
            axis.move(direction * step)
            gradient = feedback()
            point_reached = maximizer.set_point_reached(gradient)

            if axis.hard_position_limit_reached() or \
               point_reached or \
               not maximizer.is_better(gradient):
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
                      (gradient, str(axis.position)))
        except LimitError as e:
            direction, step = turn(direction, step)

    log.info("Maximum gradient: %g found at position: %s" %
             (gradient, str(axis.position)))


class Focuser(ConcertObject):
    def __init__(self, axis, epsilon, gradient_feedback):
        self._axis = axis
        self._epsilon = epsilon
        self._gradient_feedback = gradient_feedback
        self._register_message("focus")

    def focus(self, step, blocking=False):
        """A simple focusing process using gradient as a measure. It maximizes
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
        params = (self._axis, self._gradient_feedback, step, self._epsilon)
        return self._launch("focus", _focus, params, blocking)
