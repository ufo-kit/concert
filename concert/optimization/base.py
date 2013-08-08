"""Optimization base class for executing various algorithms."""

from concert.processes.base import Process
from concert.asynchronous import async, dispatcher
import logbook
import numpy as np
from concert.optimization import algorithms


class Optimizer(Process):

    """
    Base optimizer class. All necessary parameters are handled by it.
    The subclasses then implement their :py:meth:`is_better` methods,
    where an old value and new value are compared.
    """
    FOUND = "optimum-found"

    def __init__(self, param, feedback, step, algorithm=algorithms.halve,
                 max_iterations=100, epsilon=0.01):
        super(Optimizer, self).__init__([param])
        self.param = param
        self.feedback = feedback
        self.algorithm = algorithm
        self.step = step
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.value = None
        self._logger = logbook.Logger(__name__ + "." + self.__class__.__name__)

    @async
    def run(self):
        """Run the optimization algorithm."""
        # Since step is a quantity, make a safe copy here for not changing
        # the original value (we want to reuse it by the next run.
        data = self.algorithm(self.param, self.feedback, self.cmp_set,
                              np.copy(self.step) * self.step.units,
                              self.epsilon,
                              self.max_iterations)

        if len(data) < self.max_iterations:
            dispatcher.send(self, self.FOUND)

        return data

    def cmp_set(self, new_value):
        """
        Return if the *new_value* is better than current value. Also set
        the object's value to be the *new_value*.
        """
        is_better = self.is_better(new_value)
        self.value = new_value

        return is_better

    def is_better(self, value):
        """Return if the *value* is better than current value."""
        raise NotImplementedError
