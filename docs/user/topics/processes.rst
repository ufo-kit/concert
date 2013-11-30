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


Focusing
========

To adjust the focal plane of a camera, you use :func:`.focus` like this::

    from concert.processes import focus
    from concert.cameras.uca import Camera
    from concert.motors.ankatango import Motor

    motor = Motor(tango_device)
    camera = Camera('pco')
    focus(camera, motor)


Coroutine-based processing
==========================

In Concert there are *generators* which represent the source of data
and can also be used as normal iterators, e.g. in a ``for`` loop. *generators*
are not coroutines but they form a basis for them. There are two types of
coroutines in Concert, *filters* and *sinks*. *filters* get data, process it
and send them forward. Their first argument is always another coroutine to
which we send the processed data. Processing nodes which do not forward
anything are called *sinks*, e.g. a file writer.


Connection data sources with coroutines
---------------------------------------

In order to connect a *generator* that ``yields`` data to a *filter* or a
*sink* it is necessary to bootstrap the pipeline by using the
:py:func:`.inject` function, which forwards generated data to a coroutine::

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


High-performance processing
---------------------------

The generators and coroutines yield execution, but if the data production
should not be stalled by data consumption the coroutine should only provide
data buffering and delegate the real consumption to a separate thread or
process. The same can be achieved by first buffering the data and then
yielding them by a generator. It comes from the fact that a generator
will not produce a new value until the old one has been consumed.



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
