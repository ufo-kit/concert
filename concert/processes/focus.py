'''
Created on Mar 13, 2013

@author: farago
'''
import logging
from concert.optimization.scalar import Maximizer
from threading import Thread
from concert.base import ConcertObject


class FocuserMessage(object):
    FOCUS_FOUND = "focus_found"


class Focuser(ConcertObject):
    def __init__(self, axis, epsilon, gradient_feedback):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._axis = axis
        # A function which provides gradient feedback.
        self._gradient_feedback = gradient_feedback
        self._maximizer = Maximizer(epsilon)

    def _turn(self, direction, step):
        return -direction, step / 2.0

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
        def _focus(step):
            direction = 1
            hits = 0
            while True:
                self._axis.move(direction*step, True)
                gradient = self._gradient_feedback()
                point_reached = self._maximizer.set_point_reached(gradient)
                if self._axis.hard_position_limit_reached() or\
                        point_reached or\
                        not self._maximizer.is_better(gradient):
                    direction, step = self._turn(direction, step)
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
                self._maximizer.value = gradient
                self._logger.debug("Gradient: %g, axis position: %s" %
                                   (gradient, str(self._axis.get_position())))
            self._logger.info("Maximum gradient: %g found at position: %s" %
                              (gradient, str(self._axis.get_position())))
            self.send(FocuserMessage.FOCUS_FOUND)

        if blocking:
            _focus(step)
        else:
            t = Thread(target=_focus, args=(step,))
            t.daemon = True
            t.start()
