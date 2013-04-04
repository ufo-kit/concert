"""
The :mod:`.scan` module provides process functions to change parameters along
pre-computed "trajectories". For example, you can use :func:`ascan` to move a
set of motors in intervals similar to SPEC's ascan ::

    import quantities as q
    from concert.processes.scan import ascan

    def do_something(parameters):
        for each parameter in parameters:
            print(parameter)

    ascan([(motor1['position'], 0 * q.mm, 25 * q.mm),
           (motor2['position'], -2 * q.cm, 4 * q.cm)],
           n_intervals=10, handler=do_something)
"""
from concurrent.futures import wait


def _pull_first(tuple_list):
    for tup in tuple_list:
        yield tup[0]


def ascan(param_list, n_intervals, handler, initial_values=None):
    """
    For each of the *n_intervals* and for each of the *(parameter, start,
    stop)* tuples in *param_list*, calculate a set value from *(stop - start) /
    n_intervals* and set *parameter* to it::

        ascan([(motor['position'], 0 * q.mm, 2 * q.mm)], 5, handler)

    When all devices have reached the set point *handler* is called with a list
    of the parameters as its first argument.

    If *initial_values* is given, it must be a list with the same length as
    *devices* containing start values from where each device is scanned.
    """
    parameters = [param for param in _pull_first(param_list)]

    if initial_values:
        if len(param_list) != len(initial_values):
            raise ValueError("*initial_values* must match *parameter_list*")
    else:
        initial_values = []

        for param in parameters:
            if param.unit:
                initial_values.append(0 * param.unit)
            else:
                initial_values.append(0)

    initialized_params = map(lambda (tup, single): tup + (single,),
                             zip(param_list, initial_values))

    for i in range(n_intervals + 1):
        futures = []

        for param, start, stop, init in initialized_params:
            step = (stop - start) / n_intervals
            value = init + start + i * step
            futures.append(param.set(value))

        wait(futures)
        handler(parameters)


def dscan(parameter_list, n_intervals, handler):
    """
    For each of the *n_intervals* and for each of the *(parameter, start,
    stop)* tuples in *param_list*, calculate a set value from *(stop - start) /
    n_intervals* and set *parameter*.
    """
    initial_values = [param.get().result()
                      for param in _pull_first(parameter_list)]

    return ascan(parameter_list, n_intervals, handler, initial_values)
