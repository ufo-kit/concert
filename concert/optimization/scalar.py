"""Scalar Optimizers."""

from concert.optimization.base import Optimizer
from concert.optimization import algorithms


class Maximizer(Optimizer):

    """Maximize a parameter."""

    def __init__(self, param, feedback, step, algorithm=algorithms.halve,
                 max_iterations=100, epsilon=0.01):
        super(Maximizer, self).__init__(param, feedback, step, algorithm,
                                        max_iterations, epsilon)

    def is_better(self, value):
        """Return if the *value* is greater than current value."""
        return value > self.value


class Minimizer(Optimizer):

    """Minimize a parameter."""

    def __init__(self, param, feedback, step, algorithm=algorithms.halve,
                 max_iterations=100, epsilon=0.01):
        super(Minimizer, self).__init__(param, feedback, step, algorithm,
                                        max_iterations, epsilon)

    def is_better(self, value):
        """Return if the *value* is smaller than current value."""
        return value < self.value
