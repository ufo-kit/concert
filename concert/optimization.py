"""
Optimization is a procedure to iteratively find the best possible match
to

.. math::
    y = f(x).

This module provides execution routines and algorithms for optimization.
"""
import logging
from functools import wraps
from concert.coroutines.base import background, run_in_executor, run_in_loop_thread_blocking
from concert.quantities import q


LOG = logging.getLogger(__name__)


@background
async def optimize(function, x_0, algorithm, alg_args=(), alg_kwargs=None,
                   callback=None):
    """
    optimize(function, x_0, algorithm, alg_args=(), alg_kwargs=None, callback=None)

    Optimize y = await *function* (x), so *function* must be a coroutine function. *x_0* is the
    initial guess. *algorithm* is the optimization algorithm to be used::

        algorithm(x_0, *alg_args, **alg_kwargs)

    *callback* is a callable called with all the (x, y) values as they are obtained.
    """
    alg_kwargs = {} if alg_kwargs is None else alg_kwargs
    data = []

    async def evaluate(x_val):
        """Execute y = f(*x*), save (x, y) pair and return y."""
        result = await function(x_val)
        pair = x_val, result
        if callback:
            await callback(pair)
        data.append(pair)

        return result

    result = await algorithm(evaluate, x_0, *alg_args, **alg_kwargs)

    return result, data


@background
async def optimize_parameter(parameter, feedback, x_0, algorithm, alg_args=(),
                             alg_kwargs=None, callback=None):
    """
    Optimize *parameter* and use the *feedback* (a coroutine function)
    as a result. Other arguments are the same as by :func:`optimize`.
    The function to be optimized is determined as follows::

        await parameter.set(x)
        y = await feedback()

    *callback* is the same as by :func:`optimize`.
    """
    async def function(x_val):
        """
        Create a function from setting a parameter and getting a feedback.
        """
        # Do not go out of the limits
        if parameter.lower is not None:
            if parameter.lower is not None and x_val < parameter.lower:
                x_val = parameter.lower
        if parameter.upper is not None:
            if parameter.upper is not None and x_val > parameter.upper:
                x_val = parameter.upper
        await parameter.set(x_val)

        return await feedback()

    return await optimize(function, x_0, algorithm, alg_args=alg_args,
                          alg_kwargs=alg_kwargs, callback=callback)


@background
async def halver(function, x_0, initial_step=None, epsilon=None,
                 max_iterations=100):
    """
    Halving the interval, evaluate *function* based on *param*. Use
    *initial_step*, *epsilon* precision and *max_iterations*.
    """
    if initial_step is None:
        # Figure out the step based on x_0 units (take one in the given unit)
        step = q.Quantity(1, x_0.units)
    else:
        step = initial_step
    if epsilon is None:
        epsilon = q.Quantity(1e-4, x_0.units)

    direction = 1
    i = 0
    last_x = x_0

    y_0 = await function(x_0)
    # Remember the best x and y
    best = x_0, y_0

    def turn(direction, step):
        """Turn to opposite direction and reduce the step by half."""
        return -direction, step / 2

    def move(x_0, direction, step):
        """Move to a *direction* by a *step*."""
        return x_0 + direction * step

    x_0 = move(x_0, direction, step)

    while i < max_iterations:
        y_1 = await function(x_0)

        if step < epsilon:
            break

        if y_1 >= y_0:
            # Worse, change direction and move to the half of the last
            # good x and the new x.
            direction, step = turn(direction, step)
            x_0 = (x_0 + last_x) / 2
        else:
            # OK, move forward.
            if y_1 < best[1]:
                # The new y is better then the so far obtained one
                best = x_0, y_1
            last_x = x_0
            x_0 = move(x_0, direction, step)

        y_0 = y_1
        i += 1

    # Apply the best known value
    await function(best[0])

    return best[0]


def _quantized(strip_func):
    """
    Decorator to quantize a *function* which does not take units into
    account. Strips intermediate results based on *strip_func* in order
    to fit *function*'s first parameter signature.
    """
    @wraps(strip_func)
    def stripped(function):
        @wraps(function)
        async def wrapper(eval_func, x_0, *args, **kwargs):
            def q_func(x):
                """We need to run the objective function in a separate thread and loop because we
                have been called from a running event loop and have things to do in an event loop
                and can't use the running one.
                """
                coro = eval_func(q.Quantity(strip_func(x), x_0.units))
                return run_in_loop_thread_blocking(coro)

            def to_run_in_exec():
                """Run the optimization *function* in a pool so that we comply with the async nature
                of concert.
                """
                return function(q_func, x_0.magnitude, *args, **kwargs)

            dim_less = await run_in_executor(to_run_in_exec)
            return q.Quantity(dim_less, x_0.units)
        return wrapper

    return stripped


@_quantized(lambda x: x[0])
def down_hill(function, x_0, **kwargs):
    """
    down_hill(function, x_0, **kwargs)

    Downhill simplex algorithm from :py:func:`scipy.optimize.fmin`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize as scipy_optimize

    return scipy_optimize.fmin(function, x_0, disp=0, **kwargs)[0]


@_quantized(lambda x: x[0])
def powell(function, x_0, **kwargs):
    """
    powell(function, x_0, **kwargs)

    Powell's algorithm from :py:func:`scipy.optimize.fmin_powell`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize as scipy_optimize

    return scipy_optimize.fmin_powell(function, x_0, disp=0, **kwargs)


@_quantized(lambda x: x[0])
def nonlinear_conjugate(function, x_0, **kwargs):
    """
    nonlinear_conjugate(function, x_0, **kwargs)

    Nonlinear conjugate gradient algorithm from
    :py:func:`scipy.optimize.fmin_cg`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize as scipy_optimize

    return scipy_optimize.fmin_cg(function, x_0, disp=0, **kwargs)[0]


@_quantized(lambda x: x[0])
def bfgs(function, x_0, **kwargs):
    """
    bfgs(function, x_0, **kwargs)

    Broyde-Fletcher-Goldfarb-Shanno (BFGS) algorithm from
    :py:func:`scipy.optimize.fmin_bfgs`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize as scipy_optimize

    return scipy_optimize.fmin_bfgs(function, x_0, disp=0, **kwargs)[0]


@_quantized(lambda x: x[0])
def least_squares(function, x_0, **kwargs):
    """
    least_squares(function, x_0, **kwargs)

    Least squares algorithm from :py:func:`scipy.optimize.leastsq`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize as scipy_optimize

    return scipy_optimize.leastsq(function, x_0, **kwargs)[0][0]
