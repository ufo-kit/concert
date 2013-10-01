===============
Process control
===============

Scanning
========

:func:`scan` is used to scan a device parameter and start a feedback action.
For instance, to set 10 motor positions between 5 and 12 millimeter and acquire
the flow rate of a pump could be written like::

    from concert.processes import scan

    # Assume motor and pump are already defined

    def get_flow_rate():
        return pump.flow_rate

    x, y = scan(motor['position'], get_flow_rate,
                5*q.mm, 12*q.mm, 10).result()

As you can see :func:`scan` always yields a future that needs to be resolved
when you need the result.

:func:`ascan` and :func:`dscan` are used to scan multiple parameters
in a similar way as SPEC::

    from concert.quantities import q
    from concert.processes import ascan

    def do_something(parameters):
        for each parameter in parameters:
            print(parameter)

    ascan([(motor1['position'], 0 * q.mm, 25 * q.mm),
           (motor2['position'], -2 * q.cm, 4 * q.cm)],
           n_intervals=10, handler=do_something)

.. automodule:: concert.processes
    :members:


Optimization
============

This module provides various algorithms for optimizing function y = f(x)
and routines for executing the optimization.

.. automodule:: concert.optimization
    :members:

