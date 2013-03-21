"""
The :mod:`.scan` module provides process functions to move motors on
pre-computed trajectories. Similar to SPEC there is :func:`ascan` to move a set
of motors in intervals ::

    import quantities as q
    from concert.processes.scan import ascan

    ascan([(axis1, 0 * q.mm, 25 * q.mm),
           (axis2, -2 * q.cm, 4 * q.cm)],
          10)

To be notified when an axis reaches a chosen position, you can do ::

    def on_axis1_position(axis):
        print("axis1 reached position " + str(axis.get_position))

    def on_axis2_position(axis):
        print("axis2 reached position " + str(axis.get_position))

    axis1.subscribe('position', on_axis1_position)
    axis2.subscribe('position', on_axis2_position)
"""
from concert.base import launch, wait, MultiContext


def ascan(axes, intervals, blocking=False):
    """
    For each *(axis, start, stop)* in the *axes* list, move the axis in
    interval steps of *(stop - start) / intervals*.

    Each axis moves the same number of intervals, totalling in *intervals + 1*
    data points.
    """
    def do_ascan():
        axes_list = [tup[0] for tup in axes]

        with MultiContext(axes_list):
            for i in range(intervals + 1):
                events = []

                for axis, start, stop in axes:
                    step = (stop - start) / intervals
                    position = start + i * step
                    events.append(axis.set_position(position))

                wait(events)

    return launch(do_ascan, (), blocking)
