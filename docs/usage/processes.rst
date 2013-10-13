===============
Process control
===============

Scanning
========

:func:`.scan` is used to scan a device parameter and start a feedback action.
For instance, to set 10 motor positions between 5 and 12 millimeter and acquire
the flow rate of a pump could be written like::

    from concert.processes import scan

    # Assume motor and pump are already defined

    def get_flow_rate():
        return pump.flow_rate

    x, y = scan(motor['position'], get_flow_rate,
                5*q.mm, 12*q.mm, 10).result()

As you can see :func:`.scan` always yields a future that needs to be resolved
when you need the result.

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

.. autofunction:: concert.processes.scan
.. autofunction:: concert.processes.ascan
.. autofunction:: concert.processes.dscan


Focusing
========

To adjust the focal plane of a camera, you use :func:`.focus` like this::

    from concert.processes import focus
    from concert.cameras.uca import Camera
    from concert.motors.ankatango import Motor

    motor = Motor(tango_device)
    camera = Camera('pco')
    focus(camera, motor)

.. autofunction:: concert.processes.focus


Optimization
============

This module provides various algorithms for optimizing function y = f(x)
and routines for executing the optimization.

.. automodule:: concert.optimization
    :members:


Coroutine-based processing
==========================

Coroutines provide a way to process data and yield execution until more data
is produced. To build flexible coroutine-based processing pipelines in Python,
the enhanced ``yield`` statement is used. To simplify startup of the coroutine,
you can decorate a function with :py:func:`.coroutine`::

    from concert.helpers import coroutine

    @coroutine
    def printer():
        while True:
            item = yield
            print(item)

Because the ``printer`` only consumes data it is an end point, hence called a
sink. Filters on the other hand hook into the stream and turn the input into
some output. For example, to generate a stream of squared input, you would
write::

    @coroutine
    def square(target):
        while True:
            item = yield
            target.send(item**2)

.. autofunction:: concert.helpers.coroutine


Connection data sources with coroutines
---------------------------------------

There are two ways to produce data for a coroutine. The recommended way is to
write a function that sends data *into* a coroutine::

    def source(n, target):
        for i in range(n):
            target.send(i)

This inserts the numbers 0 to n-1 into the coroutine ``target``. To connect
``source`` with the ``printer`` coroutine, you simply call the coroutine as the
argument of the source::

    source(5, printer())

In some cases, you will be faced with a generator that ``yields`` data instead
of sending it. In that case, use the :py:func:`.inject` function to forward
generated data to a coroutine::

    from concert.helpers import inject

    def generator(n):
        for i in range(n):
            yield i

    inject(generator(5), printer())

To fan out a single input stream to multiple consumers, you can use the
:py:func:`.broadcast` like this::

    from concert.helpers import broadcast

    source(5, broadcast(printer(),
                        square(printer())))

.. autofunction:: concert.helpers.inject
.. autofunction:: concert.helpers.broadcast


High-performance processing
---------------------------

The generators and coroutines yield execution, but if the data production
should not be stalled by data consumption the coroutine should only provide
data buffering and delegate the real consumption to a separate thread or
process. The same can be achieved by first buffering the data and then
yielding them by a generator. It comes from the fact that a generator
will not produce a new value until the old one has been consumed.


Pre-defined coroutines
----------------------

.. automodule:: concert.coroutines
    :members:


Data processing with Ufo
========================

The :mod:`.ufo` module provides classes to process data from an experiment with
the UFO data processing framework. The simplest example could look like this::

    from concert.ext.ufo import InjectProcess
    from gi.repository import Ufo
    import numpy as np
    import scipy.misc

    pm = Ufo.PluginManager()
    writer = pm.get_task('writer')
    writer.props.filename = 'foo-%05i.tif'

    proc = InjectProcess(writer)

    proc.run()
    proc.push(scipy.misc.lena())
    proc.wait()


To save yourself some time, the :mod:`.ufo` module provides a wrapper around the raw ``UfoPluginManager``::

    from concert.ext.ufo import PluginManager

    pm = PluginManager()
    writer = pm.get_task('writer', filename='foo-%05i.tif')

.. automodule:: concert.ext.ufo
    :members:
