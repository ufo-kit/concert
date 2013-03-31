"""
The :mod:`.scan` module provides process functions to move motors on
pre-computed trajectories. Similar to SPEC there is :func:`ascan` to move a set
of motors in intervals ::

    import quantities as q
    from concert.processes.scan import ascan

    ascan([(motor1, 0 * q.mm, 25 * q.mm),
           (motor2, -2 * q.cm, 4 * q.cm)],
          10)

To be notified when an motor reaches a chosen position, you can do ::

    def on_motor1_position(motor):
        print("motor1 reached position " + str(motor.get_position))

    def on_motor2_position(motor):
        print("motor2 reached position " + str(motor.get_position))

    motor1.subscribe('position', on_motor1_position)
    motor2.subscribe('position', on_motor2_position)
"""
from concurrent.futures import wait


def _pull_first(tuple_list):
    for tup in tuple_list:
        yield tup[0]


def ascan(parameter_list, n_intervals, handler, initial_values=None):
    """
    For each *(parameter, start, stop)* in *parameter_list*, call
    ``device.set(parameter, x)`` where ``x`` is an interval *(stop - start) /
    n_intervals* ::

        ascan([(motor['position'], 0 * q.mm, 2 * q.mm)], 5)

    Each device gets the same number of intervals, totalling in *n_intervals +
    1* data points.

    If *initial_values* is given, it must be a list with the same length as
    *devices* containing start values from where each device is scanned.
    """
    parameters = [param for param in _pull_first(parameter_list)]

    def do_ascan(initial_values):
        initialized_params = map(lambda (tup, single): tup + (single,),
                                 zip(parameter_list, initial_values))

        for i in range(n_intervals + 1):
            futures = []

            for param, start, stop, init in initialized_params:
                step = (stop - start) / n_intervals
                value = init + start + i * step
                futures.append(param.set(value))

            wait(futures)
            handler(parameters)

    if initial_values:
        if len(parameter_list) != len(initial_values):
            raise ValueError("*initial_values* must match *parameter_list*")
    else:
        initial_values = [0 * param.unit for param in parameters]

    do_ascan(initial_values)


def dscan(parameter_list, n_intervals, handler):
    """
    For each *(motor, start, stop)* in the *devices* list, move the motor in
    interval steps of *(stop - start) / intervals* relative to the motors'
    initial position.

    Each motor moves the same number of intervals, totalling in *intervals + 1*
    data points.
    """
    initial_values = [param.get().result()
                      for param in _pull_first(parameter_list)]

    return ascan(parameter_list, n_intervals, handler, initial_values)
