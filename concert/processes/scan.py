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


def ascan(motors, n_intervals, start_positions=None, blocking=False):
    """
    For each *(motor, start, stop)* in the *motors* list, move the motor in
    interval steps of *(stop - start) / n_intervals*.

    Each motor moves the same number of intervals, totalling in *n_intervals +
    1* data points.

    If *start_positions* is given, it must be a list with the same length as
    *motors* containing start positions from where scanning for each motor
    should begin.
    """
    def do_ascan(start_positions=None):
        motors_list = [tup[0] for tup in motors]

        if start_positions:
            if len(motors) != len(start_positions):
                raise ValueError("*start_positions* must match *motors*")
        else:
            start_positions = [0 * q.mm] * len(motors)

        positioned_motors = map(lambda (tup, single): tup + (single,),
                                zip(motors, start_positions))

        with MultiContext(motors_list):
            for i in range(n_intervals + 1):
                events = []

                for motor, start, stop, init in positioned_motors:
                    step = (stop - start) / n_intervals
                    position = init + start + i * step
                    events.append(motor.set_position(position))

                wait(events)

    return launch(do_ascan, (start_positions,), blocking)


def dscan(motors, n_intervals, blocking=False):
    """
    For each *(motor, start, stop)* in the *motors* list, move the motor in
    interval steps of *(stop - start) / intervals* relative to the motors'
    initial position.

    Each motor moves the same number of intervals, totalling in *intervals + 1*
    data points.
    """
    start_positions = [m[0].get_position() for m in motors]
    return ascan(motors, n_intervals, start_positions, blocking)
