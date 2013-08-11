import logbook
from concert.optimization.base import ParameterOptimizer
from concert.base import LimitError


LOG = logbook.Logger()


class Minimizer(ParameterOptimizer):

    """Minimizer tries to minimize a function

    .. math::
        y = f(x)

    .. py:attribute:: algorithm

        An algorithm which does the optimization, it is a callable.

    .. py:attribute:: alg_args

        A tuple of arguments passed to the algorithm

    .. py:attribute:: alg_kwargs

        A dictionary of keyword arguments passed to the algorithm

    The executive optimization function is then::

        algorithm(x_guess, *alg_args, **alg_kwargs)

    where *x_guess* is the x value at which the optimizer starts. If
    *alg_args* is None, the *x_guess* is derived from the current
    parameter value, otherwise *x_guess* must be the first value in
    the *alg_args* list.
    """

    def __init__(self, param, feedback, algorithm, alg_args=None,
                 alg_kwargs=None):
        super(Minimizer, self).__init__(param, feedback)
        self.algorithm = algorithm
        self.alg_args = alg_args
        if not self.alg_args:
            self.alg_args = (param.get().result(), )
        self.alg_kwargs = alg_kwargs
        if not self.alg_kwargs:
            self.alg_kwargs = {}

    def _optimize(self):
        result = self.algorithm(self.evaluate, *self.alg_args,
                                **self.alg_kwargs)
        try:
            self.param.set(result).wait()
        except LimitError:
            LOG.debug("Limit reached.")


class Maximizer(Minimizer):

    """
    The same as the :py:class:`.Minimizer` but with changed sign of the
    feedback, that is, if the function to minimize is

    .. math::

        y = f(x)

    then the new function to maximize is

    .. math::

        y = - f(x).
    """

    def __init__(self, param, feedback, algorithm, alg_args=None,
                 alg_kwargs=None):
        super(Maximizer, self).__init__(param, lambda: -feedback(),
                                        algorithm, alg_args, alg_kwargs)
