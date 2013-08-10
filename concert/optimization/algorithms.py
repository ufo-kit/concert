import numpy as np


def halver(function, param, initial_step=None, epsilon=None,
           max_iterations=100):
    """
    Halving the interval, evaluate *function* based on *param*. Use
    *initial_step*, *epsilon* precision and *max_iterations*.
    """
    # Safe copy for not changing the original.
    if initial_step is None:
        step = 1 * param.get().result().units
    else:
        step = np.copy(initial_step) * initial_step.units
    if epsilon is None:
        epsilon = 1e-3 * param.get().result().units
    direction = 1
    i = 0
    y_0 = function(param.get().result())

    def turn(direction, step):
        return -direction, step / 2.0

    while i < max_iterations:
        y_1 = function(param.get().result() + direction * step)
        point_reached = step < epsilon

        if point_reached:
            break

        in_limits = True if param.limiter is None or \
            param.limiter(param.get().result() + direction * step) else False

        if y_1 >= y_0 or not in_limits:
            direction, step = turn(direction, step)
        y_0 = y_1
        i += 1

    return param.get().result()


def quantized(function):
    """Quantize a *function* which does not take units into account."""
    def wrapper(eval_func, x_0, *args, **kwargs):
        return function(lambda x: eval_func(x * x_0.units),
                        x_0, *args, **kwargs)

    wrapper.__doc__ = function.__doc__

    return wrapper


@quantized
def down_hill(function, x_0, **kwargs):
    """Downhill simplex algorithm from :py:func:`scipy.optimize.fmin`."""
    from scipy import optimize

    return optimize.fmin(function, x_0, **kwargs)[0] * x_0.units


@quantized
def powell(function, x_0, **kwargs):
    """Powell's algorithm from :py:func:`scipy.optimize.fmin_powell`."""
    from scipy import optimize

    return optimize.fmin_powell(function, x_0, **kwargs) * x_0.units


@quantized
def nonlinear_conjugate(function, x_0, **kwargs):
    """
    Nonlinear conjugate gradient algorithm from
    :py:func:`scipy.optimize.fmin_cg`.
    """
    from scipy import optimize

    return optimize.fmin_cg(function, x_0, **kwargs)[0] * x_0.units


@quantized
def bfgs(function, x_0, **kwargs):
    """
    Broyde-Fletcher-Goldfarb-Shanno (BFGS) algorithm from
    :py:func:`scipy.optimize.fmin_bfgs`.
    """
    from scipy import optimize

    return optimize.fmin_bfgs(function, x_0, **kwargs)[0] * x_0.units


@quantized
def least_squares(function, x_0, **kwargs):
    """Least squares algorithm from :py:func:`scipy.optimize.leastsq`."""
    from scipy import optimize

    return optimize.leastsq(function, x_0, **kwargs)[0][0] * x_0.units
