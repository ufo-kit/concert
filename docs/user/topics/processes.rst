===============
Process control
===============

Scanning
========

:func:`.scan` is used to scan a device parameter and start a feedback action.
For instance, to set 10 motor positions between 5 and 12 millimeter and acquire
the flow rate of a pump could be written like::

    from concert.processes.common import scan
    from concert.helpers import Region

    # Assume motor and pump are already defined

    def get_flow_rate():
        return pump.flow_rate

    # A parameter object encapsulated with its scanning positions
    region = Region(motor['position'], np.linspace(5, 12, 10) * q.mm)

    generator = scan(get_flow_rate, region)

The parameter is first wrapped into a :class:`concert.helpers.Region` object
which holds the parameter and the scanning region for parameters. :func:`.scan`
is multidimensional, i.e. you can scan as many parameters as you need, from 1D
scans to complicated multidimensional scans. If you want to scan just one
parameter, pass the region instance, if you want to scan more, pass a list or
tuple of region instances. :func:`.scan` returns a generator which yields
futures. This way the scan is asynchronous and you can continuously see its
progress by resolving the yielded futures. Each future then returns the result
of one iteration as tuples, which depends on how many parameters scan gets on
input (scan dimensionality). The general signature of future results is *(x_0,
x_1, ..., x_n, y)*, where *x_i* are the scanned parameter values and *y* is the
result of *feedback*. For resolving the futures you would use
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

A two-dimensional scan with *region_2* parameter in the inner (fastest changing)
loop could look as follows::

    region_1 = Region(motor_1['position'], np.linspace(5, 12, 10) * q.mm)
    region_2 = Region(motor_2['position'], np.linspace(0, 10, 5) * q.mm)

    generator = scan(get_flow_rate, [region_1, region_2])

You can set callbacks which are called when some parameter is changed during a
scan. This can be useful when you e.g. want to acquire a flat field when the
scan takes a long time. For example, to acquire tomograms with different
exposure times and flat field images you can do::


    import numpy as np
    from concert.async import resolve
    from concert.helpers import Region

    def take_flat_field():
        # Do something here
        pass

    exp_region = Region(camera['exposure_time'], np.linspace(1, 100, 100) * q.ms)
    position_region = Region(motor['position'], np.linspace(0, 180, 1000) * q.deg)
    callbacks = {exp_region: take_flat_field}

    # This is a 2D scan with position_region in the inner loop. It acquires a tomogram, changes
    # the exposure time and continues like this until all exposure times are exhausted.
    # Take_flat_field is called every time the exposure_time of the camera is changed
    # (in this case after every tomogram) and you can use it to correct the acquired images.
    for result in resolve(scan(camera.grab, [exp_region, position_region], callbacks=callbacks)):
        # Do something real instead of just a print
        print result


:func:`.ascan` and :func:`.dscan` are used to scan multiple parameters
in a similar way as SPEC::

    from concert.quantities import q
    from concert.processes.common import ascan

    def do_something(parameters):
        for each parameter in parameters:
            print(parameter)

    ascan([(motor1['position'], 0 * q.mm, 25 * q.mm),
           (motor2['position'], -2 * q.cm, 4 * q.cm)],
           n_intervals=10, handler=do_something)


Focusing
========

To adjust the focal plane of a camera, you use :func:`.focus` like this::

    from concert.processes.common import focus
    from concert.cameras.uca import Camera
    from concert.motors.dummy import LinearMotor

    motor = LinearMotor()
    camera = Camera('mock')
    focus(camera, motor)
