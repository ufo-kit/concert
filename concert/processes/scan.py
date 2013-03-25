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
import quantities as q
from concert.base import launch, wait, MultiContext


def ascan(devices, n_intervals, initial_values=None, blocking=False):
    """
    For each *(device, parameter, start, stop)* in the *devices* list, call
    ``device.set(parameter, x)`` where ``x`` is an interval *(stop - start) /
    n_intervals*.

    Each device gets the same number of intervals, totalling in *n_intervals +
    1* data points.

    If *initial_values* is given, it must be a list with the same length as
    *devices* containing start values from where each device is scanned.
    """
    def do_ascan(initial_values):
        device_list = [tup[0] for tup in devices]

        positioned_devices = map(lambda (tup, single): tup + (single,),
                                 zip(devices, initial_values))

        with MultiContext(device_list):
            for i in range(n_intervals + 1):
                events = []

                for device, param, start, stop, init in positioned_devices:
                    step = (stop - start) / n_intervals
                    value = init + start + i * step
                    events.append(device.set(param, value))

                wait(events)

    if initial_values:
        if len(devices) != len(initial_values):
            raise ValueError("*start_positions* must match *motors*")
    else:
        initial_values = [0 * q.mm] * len(devices)

    return launch(do_ascan, (initial_values,), blocking)


def dscan(devices, n_intervals, blocking=False):
    """
    For each *(motor, start, stop)* in the *devices* list, move the motor in
    interval steps of *(stop - start) / intervals* relative to the motors'
    initial position.

    Each motor moves the same number of intervals, totalling in *intervals + 1*
    data points.
    """
    initial_values = [d[0].get(d[1]) for d in devices]
    return ascan(devices, n_intervals, initial_values, blocking)
