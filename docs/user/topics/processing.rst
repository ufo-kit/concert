===============
Data processing
===============

Coroutines
==========

Coroutines provide a way to process data and yield execution until more data is
produced. *Generators* represent the source of data and can be used as normal
iterators, e.g.  in a ``for`` loop. Coroutines can use the output of a generator
to either process data and output a new result item in a *filter* fashion or
process the data without further results in a *sink* fashion. For more on
coroutines, see :ref:`concurrent-execution`.

Data processing with coroutines uses generators and iterates over their items
like this::

    async def producer(num):
        for i in range(num):
            yield i

    async def printer(producer):
        async for item in producer:
            print(item)

    # Usage:
    await printer(producer(10))

*printer* coroutine fetches data items and prints them one by one. Because no data is
produced, this coroutine falls into the sink category. Concert provides some
common pre-defined sinks in the :mod:`.sinks` module.

Filters hook into the data stream and process the input to produce some output.
For example, to generate a stream of squared input, you would write::

    def square(consumer):
        async for item in producer:
            yield item ** 2

    # Usage:
    await printer(square(producer(10)))

You can find a variety of pre-defined filters in the :mod:`.filters` module.


Broadcasting
------------

To fan out a single input stream to multiple consumers, you can use the
:func:`.broadcast`. Its first argument is the producer and the rest are
consumers. :func:`.broadcast` creates the connections from producer to consumers
and returns a list of coroutines, which can be used by :func:`asyncio.gather`
function, like this::

    from concert.coroutines.base import broadcast

    coros = broadcast(producer(10), printer, printer)
    await asyncio.gather(*coros)


High-performance processing
---------------------------

The generators and coroutines yield execution, but if the data production should
not be stalled by data consumption the coroutine should only provide data
buffering and delegate the real consumption to a separate thread or process. The
same can be achieved by first buffering the data and then yielding them by a
generator. It comes from the fact that a generator will not produce a new value
until the old one has been consumed.



High-performance computing
==========================

The :mod:`.ufo` module provides classes to process data from an experiment with
the UFO data processing framework. The simplest example could look like this::

    from concert.ext.ufo import InjectProcess
    from gi.repository import Ufo
    import numpy as np
    import scipy.misc

    pm = Ufo.PluginManager()
    writer = pm.get_task('write')
    writer.props.filename = 'foo-%05i.tif'

    proc = InjectProcess(writer)

    proc.start()
    await proc.insert(scipy.misc.ascent())
    proc.wait()


To save yourself some time, the :mod:`.ufo` module provides a wrapper around the
raw ``UfoPluginManager``::

    from concert.ext.ufo import PluginManager

    pm = PluginManager()
    writer = pm.get_task('write', filename='foo-%05i.tif')



Viewing processed data
======================

Concert has a Matplotlib integration to simplify viewing 1D time series with the
:class:`.PyplotViewer`. For 2D, there are multiple implementations, for details
see :ref:`viewers` and Concert examples_.

.. _examples: https://github.com/ufo-kit/concert-examples/blob/master/pyplotimageviewer-example.py


Writing image data
==================

Concert provides :class:`.DirectoryWalker` for traversing the filesystem and
writing image sequences. You can use its :meth:`.descend` method to descend into
a sub-directory and the :meth:`.ascend` method to return one level back.

If you just want to write images in the current directory use the
:meth:`~concert.storage.Walker.write` method. To create an image writer
in either the current directory or one level below, you can use the
:meth:`.create_writer` method. This method creates the writer and if you specify
a sub-directory also ascends back. You should use this in a `with` statement to
make sure that while you are creating the image writer, some other coroutine
does not change walker's path. The writing itself can then happen after the
`with` statement::

    async with walker:
        writer = walker.create_writer(producer, name='subdirectory')

    # create_writer ascends back so the writing itself can happen outside of the
    # with statement
    await writer
