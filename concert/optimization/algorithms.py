"""
Optimization (minimization, maximization) can be done by many techniques.
This module consists of algorithms capable of optimizing functions y = f(x).
"""

from concert.quantities import q


def halver(function, x_0, initial_step=None, epsilon=None,
           max_iterations=100):
    """
    Halving the interval, evaluate *function* based on *param*. Use
    *initial_step*, *epsilon* precision and *max_iterations*.
    """
    # Safe copy for not changing the original.
    if initial_step is None:
        step = x_0 / x_0.magnitude if x_0.magnitude else x_0
    else:
        step = initial_step
    if epsilon is None:
        epsilon = 1e-3 * x_0

        if x_0.magnitude:
            epsilon /= x_0.magnitude

    direction = 1
    i = 0
    last_x = x_0

    y_0 = function(x_0)

    def turn(direction, step):
        return -direction, step / 2.0

    def move(x_0, direction, step):
        return x_0 + direction * step

    x_0 = move(x_0, direction, step)

    while i < max_iterations:
        y_1 = function(x_0)

        if step < epsilon:
            break

        if y_1 >= y_0:
            # Worse, change direction and move to the half of the last
            # good x and the new x.
            direction, step = turn(direction, step)
            x_0 = (x_0 + last_x) / 2
        else:
            # OK, move forward.
            last_x = x_0
            x_0 = move(x_0, direction, step)

        y_0 = y_1
        i += 1

    return x_0


def quantized(function):
    """
    A helper function meant to be used as a decorator to quantize
    a *function* which does not take units into account.
    """
    def wrapper(eval_func, x_0, *args, **kwargs):
        return q.Quantity(function(lambda x:
                                   eval_func(q.Quantity(x, x_0.units)),
                                   x_0.magnitude, *args, **kwargs), x_0.units)

    wrapper.__doc__ = function.__doc__

    return wrapper


@quantized
def down_hill(function, x_0, **kwargs):
    """
    Downhill simplex algorithm from :py:func:`scipy.optimize.fmin`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize

    return optimize.fmin(function, x_0, disp=0, **kwargs)[0]


@quantized
def powell(function, x_0, **kwargs):
    """
    Powell's algorithm from :py:func:`scipy.optimize.fmin_powell`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize

    return optimize.fmin_powell(function, x_0, disp=0, **kwargs)


@quantized
def nonlinear_conjugate(function, x_0, **kwargs):
    """
    Nonlinear conjugate gradient algorithm from
    :py:func:`scipy.optimize.fmin_cg`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize

    return optimize.fmin_cg(function, x_0, disp=0, **kwargs)[0]


@quantized
def bfgs(function, x_0, **kwargs):
    """
    Broyde-Fletcher-Goldfarb-Shanno (BFGS) algorithm from
    :py:func:`scipy.optimize.fmin_bfgs`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize

    return optimize.fmin_bfgs(function, x_0, disp=0, **kwargs)[0]


@quantized
def least_squares(function, x_0, **kwargs):
    """
    Least squares algorithm from :py:func:`scipy.optimize.leastsq`.
    Please refer to the scipy function for additional arguments information.
    """
    from scipy import optimize

    return optimize.leastsq(function, x_0, **kwargs)[0][0]
