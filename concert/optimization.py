"""
Optimization is a procedure to iteratively find the best possible match
to

.. math::
    y = f(x).

This module provides execution routines and algorithms for optimization.
"""
import logging
from concert.coroutines.base import background
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
        lower = await parameter.get_lower()
        upper = await parameter.get_upper()
        if lower is not None:
            if lower is not None and x_val < lower:
                x_val = lower
        if upper is not None:
            if upper is not None and x_val > upper:
                x_val = upper
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


@background
async def scipy_minimize(func, x_0, **kwargs):
    """Use :py:func:`scipy.optimize.minimize`, *func* is a coroutine function, the translation to
    scipy is taken care of here. *x_0* is the initial guess and *kwargs* are passed to *minimize*.
    """
    import asyncio
    from threading import Thread
    from scipy.optimize import minimize

    def optimize_in_thread(loop, future):
        # This function runs in a separate thread, never use the loop in any other way than
        # abc_threadsafe.
        def fun_for_scipy(x, *args):
            # Scipy uses an array becuase "minimize" can optimize a function with more variables,
            # take just the first value and add the unit
            x_orig = q.Quantity(x[0], x_0.units)
            # Call the coroutine function to be optimized in main thread
            return asyncio.run_coroutine_threadsafe(func(x_orig, *args), loop).result()

        try:
            result = minimize(fun_for_scipy, x_0.magnitude, **kwargs)
        except Exception as exc:
            loop.call_soon_threadsafe(future.set_exception, exc)
        else:
            loop.call_soon_threadsafe(future.set_result, result)

    # Only scalar optimization supported for now
    try:
        x_0[0]
    except Exception:
        pass
    else:
        raise ValueError('Only scalar optimization is supported')

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    t = Thread(target=optimize_in_thread, args=(loop, future))
    t.start()
    try:
        result = await future
        LOG.debug("Optimization result of `%s':\n%s", func.__name__, result)
    finally:
        t.join()

    if not result.success:
        raise OptimizationError(result.message)

    return result


class OptimizationError(Exception):
    """Optimization-related errors."""
