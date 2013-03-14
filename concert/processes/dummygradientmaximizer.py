'''
Created on Mar 13, 2013

@author: farago
'''
from concert.events import type as eventtype
import logging
from concert.events.dispatcher import dispatcher
from concert.optimization.scalar import Maximizer
from threading import Thread
from concert.base import ConcertObject


class DummyGradientMaximizerState(object):
    MAXIMUM_FOUND = eventtype.make_event_id()

class DummyGradientMaximizer(ConcertObject):
    _RIGHT = 1
    def __init__(self, axis, initial_step, epsilon, gradient_feedback):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._axis = axis
        # A function which provides gradient feedback.
        self._gradient_feedback = gradient_feedback
        self._maximizer = Maximizer(1e-3)
        self._stop = False
        self._direction = DummyGradientMaximizer._RIGHT
        self._step = initial_step
        self._eps = epsilon

    def turn(self):
        self._direction = -self._direction
        self._step /= 2.0
            
    def focus(self, blocking=False):
        def _focus():
            while True:
                self._maximizer.value = self._gradient_feedback()
                self._axis.set_position(self._axis.get_position()+\
                                self._direction*self._step, blocking=True)
                if self._axis.hard_position_limit_reached() or\
                    not self._maximizer.is_better(self._gradient_feedback()):
                    self.turn()
                elif abs(self._maximizer.value - self._gradient_feedback())\
                                                            < self._eps:
                    break
                self._logger.debug("Gradient: %g, axis position: %s" %\
                   (self._gradient_feedback(), str(self._axis.get_position())))
            self._logger.info("Maximum gradient: %g found at position: %s" %\
                 (self._gradient_feedback(), str(self._axis.get_position())))
            dispatcher.send(self, DummyGradientMaximizerState.MAXIMUM_FOUND)
        
        if blocking:
            _focus()
        else:
            t = Thread(target=_focus)
            t.daemon = True
            t.start()