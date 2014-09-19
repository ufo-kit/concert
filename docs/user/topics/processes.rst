===============
Process control
===============

Scanning
========

:func:`.scan` is used to scan a device parameter and start a feedback action.
For instance, to set 10 motor positions between 5 and 12 millimeter and acquire
the flow rate of a pump could be written like::

    from concert.processes import scan
    from concert.helpers import Range

    # Assume motor and pump are already defined

    def get_flow_rate():
        return pump.flow_rate

    # A parameter object encapsulated with its scanning positions
    param_range = Range(motor['position'], 5*q.mm, 12*q.mm, 10)

    generator = scan(get_flow_rate, param_range)

The parameter is first wrapped into a :class:`concert.helpers.Range` object
which holds the parameter and the scanning range. :func:`.scan` is
multidimensional, i.e. you can scan as many parameters as you need, from 1D
scans to complicated multidimensional scans. :func:`.scan` returns a generator
which yields futures. This way the scan is asynchronous and you can continuously
see its progress by resolving the yielded futures. Each future then returns the
result of one iteration as tuples, which depends on how many parameters scan
gets on input (scan dimensionality). The general signature of future results is
*(x_0, x_1, ..., x_n, y)*, where *x_i* are the scanned parameter values and *y*
is the result of *feedback*. For resolving the futures you would use
:func:`concert.async.resolve` like this::

    from concert.async import resolve

    for tup in resolve(generator):
        # resolve yields the results of futures
        do_smth(tup)

To continuously plot the values obtained by a 1D scan by a
:class:`concert.ext.viewers.PyplotViewer` you can do::

    from concert.coroutines.base import inject
    from concert.ext.viewers import PyplotViewer

    viewer = Pyplotviewer()

    inject(resolve(generator), viewer())

A two-dimensional scan with *range_2* parameter in the inner (fastest changing)
loop could look as follows::

    range_1 = Range(motor_1['position'], 5*q.mm, 12*q.mm, 10)
    range_2 = Range(motor_2['position'], 0*q.mm, 10*q.mm, 5)

    generator = scan(get_flow_rate, range_1, range_2)


:func:`.ascan` and :func:`.dscan` are used to scan multiple parameters
in a similar way as SPEC::

    from concert.quantities import q
    from concert.processes import ascan

    def do_something(parameters):
        for each parameter in parameters:
            print(parameter)

    ascan([(motor1['position'], 0 * q.mm, 25 * q.mm),
           (motor2['position'], -2 * q.cm, 4 * q.cm)],
           n_intervals=10, handler=do_something)


Focusing
========

To adjust the focal plane of a camera, you use :func:`.focus` like this::

    from concert.processes import focus
    from concert.cameras.uca import Camera
    from concert.motors.dummy import LinearMotor

    motor = LinearMotor()
    camera = Camera('mock')
    focus(camera, motor)
