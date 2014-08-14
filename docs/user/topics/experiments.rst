===========
Experiments
===========

Experiments connect data acquisition and processing. They can be run multiple times
by the :meth:`.base.Experiment.run`, they take care of proper file structure and
logging output.


Acquisition
-----------

Experiments consist of :class:`.Acquisition` objects which encapsulate data generator
and consumers for a particular experiment part (dark fields, radiographs, ...). This
way the experiments can be broken up into smaller logical pieces. A single acquisition
object needs to be reproducible in order to repeat an experiment more times, thus we
specify its generator and consumers as callables which return the actual generator or
consumer. We need to do this because generators cannot be "restarted". An example of
an acquisition could look like this::

    from concert.coroutines.base import coroutine
    from concert.experiments import Acquisition

    # This is a real generator, num_items is provided somewhere in our session
    def produce():
        for i in range(num_items):
            yield i

    # A simple data forwarder filter, next_consumer has to be already defined
    @coroutine
    def consumer():
        while True:
            item = yield
            next_consumer.send(item)

    acquisition = Acquisition('foo', produce, consumer_callers=[consumer])
    # Now we can run the acquisition
    acquisition()

.. autoclass:: concert.experiments.base.Acquisition
    :members:


Base
----

Base :class:`.base.Experiment` makes sure a directory for each run is created and
logger output goes to that directory.

.. autoclass:: concert.experiments.base.Experiment
    :members:


Imaging
-------

Imaging experiments all subclass :class:`.imaging.Experiment`, which makes sure
all the acquired frames are written to disk.


.. autoclass:: concert.experiments.imaging.Experiment
    :members:

A basic frame acquisition generator which triggers the camera itself is provided by
:func:`.frames`

.. autofunction:: concert.experiments.imaging.frames

There are tomography helper functions which make it easier to define the proper
settings for conducting a tomographic experiment.

.. autofunction:: concert.experiments.imaging.tomo_angular_step

.. autofunction:: concert.experiments.imaging.tomo_projections_number

.. autofunction:: concert.experiments.imaging.tomo_max_speed


Control
-------

.. automodule:: concert.experiments.control
    :members:
