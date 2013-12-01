===============
Data processing
===============

Coroutines
==========

Coroutines provide a way to process data and yield execution until more data is
produced. *Generators* represent the source of data and can used as normal
iterators, e.g.  in a ``for`` loop. Coroutines can use the output of a generator
to either process data and output a new result item in a *filter* fashion or
process the data without further results in a *sink* fashion.

Coroutines are simple functions that get their input by calling yield on the
right side or as an argument. Because they need to be started in a particular
way, it is useful to decorate a coroutine with the :func:`.coroutine`
decorator::

    from concert.helpers import coroutine

    @coroutine
    def printer():
        while True:
            item = yield
            print(item)

This coroutine fetches data items and prints them one by one. Because no data is
produced, this coroutine falls into the sink category. Concert provides some
common pre-defined sinks in the :mod:`.sinks` module.

Filters hook into the data stream and process the input to produce some output.
For example, to generate a stream of squared input, you would write::

    @coroutine
    def square(consumer):
        while True:
            item = yield
            consumer.send(item**2)

You can find a variety of pre-defined filters in the :mod:`.filters` module.


Connecting data sources with coroutines
---------------------------------------

In order to connect a *generator* that ``yields`` data to a *filter* or a *sink*
it is necessary to bootstrap the pipeline by using the :func:`.inject` function,
which forwards generated data to a coroutine::

    from concert.helpers import inject

    def generator(n):
        for i in range(n):
            yield i

    # Use the output of generator to feed into printer
    inject(generator(5), printer())

To fan out a single input stream to multiple consumers, you can use the
:func:`.broadcast` like this::

    from concert.helpers import broadcast

    source(5, broadcast(printer(),
                        square(printer())))


High-performance processing
---------------------------

The generators and coroutines yield execution, but if the data production should
not be stalled by data consumption the coroutine should only provide data
buffering and delegate the real consumption to a separate thread or process. The
same can be achieved by first buffering the data and then yielding them by a
generator. It comes from the fact that a generator will not produce a new value
until the old one has been consumed.



High-performance computing with UFO
===================================

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


To save yourself some time, the :mod:`.ufo` module provides a wrapper around the
raw ``UfoPluginManager``::

    from concert.ext.ufo import PluginManager

    pm = PluginManager()
    writer = pm.get_task('writer', filename='foo-%05i.tif')
