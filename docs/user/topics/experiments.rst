===========
Experiments
===========

Experiments connect data acquisition and processing. They can be run multiple times
by the :meth:`.base.Experiment.run`, they take care of proper file structure and
logging output.


Acquisition
-----------

Experiments consist of :class:`.Acquisition` objects which encapsulate data
generator and consumers for a particular experiment part (dark fields,
radiographs, ...). This way the experiments can be broken up into smaller
logical pieces. A single acquisition object needs to be reproducible in order to
repeat an experiment more times, thus we specify its generator and consumers as
callables which return the actual generator or consumer. We need to do this
because generators cannot be "restarted".

It is very important that you enclose the executive part of the production and
consumption code in `try-finally` to ensure proper clean up. E.g. if a producer
starts rotating a motor, then in the `finally` clause there should be the call
`await motor.stop()`.

An example of an acquisition could look like this::

    from concert.experiments.base import Acquisition

    # This is a real generator, num_items is provided somewhere in our session
    async def produce():
        try:
            for i in range(num_items):
                yield i
        finally:
            # Clean up here
            pass

    # A simple coroutine sink which prints items
    async def consume(producer):
        try:
            async for item in producer:
                print(item)
        finally:
            # Clean up here
            pass

    acquisition = await Acquisition('foo', produce, consumers=[consume])
    # Now we can run the acquisition
    await acquisition()


.. autoclass:: concert.experiments.base.Acquisition
    :members:

.. autoclass:: concert.experiments.base.Consumer
    :members:


Base
----

Base :class:`.base.Experiment` makes sure all acquisitions are executed. It also
holds :class:`.addons.Addon` instances which provide some extra functionality,
e.g. live preview, online reconstruction, etc. To make a simple experiment for
running the acquisition above and storing log with
:class:`concert.storage.Walker`::

    import logging
    from concert.experiments.base import LocalAcquisition, Experiment
    from concert.storage import DirectoryWalker

    LOG = logging.getLogger(__name__)

    walker = DirectoryWalker(log=LOG)
    acquisitions = [await Acquisition('foo', produce)]
    experiment = await Experiment(acquisitions, walker)

    await experiment.run()

.. autoclass:: concert.experiments.base.Experiment
    :members:

Experiments also have a :py:attr:`.base.Experiment.log` attribute, which gets a
new handler on every experiment run and this handler stores the output in the
current experiment working directory defined by it's
:class:`concert.storage.Walker`.

Advanced
--------

Sometimes we need finer control over when exactly is the data acquired and worry
about the download later. We can use the *acquire* argument to acquisition
class. This means that the data acquisition can be invoked before data download.
Acquisition calls its *acquire* first and only when it is finished connects
producer with consumers.

The Experiment class has the attribute :py:attr:`.base.Experiment.ready_to_prepare_next_sample` which is an instance of an :class:`asyncio.Event`. This can be used to tell that most of the experiment is finished and a new iteration of
this experiment can be prepared (e.g. by the :class:`concert.directors.base.Director`.
In the :meth:`.base.Experiment.run` the :py:attr:`.base.Experiment.ready_to_prepare_next_sample` will be set that at
the end of an experiment is is always set. In the beginning of the :meth:`.base.Experiment.run` it will be cleared.
This is an example implementation making use of this::

	from concert.experiments.base import Experiment, Acquisition
	class MyExperiment(Experiment):
		async def __ainit__(self, walker, camera):
			acq = Acquisition("acquisition", self._produce_frames)
			self._camera = camera
			await super().__ainit__([acq], walker)

		async def _produce_frame(self):
			num_frames = 100
			async with self._camera.recording():
				# Do the acquisition of the frames in camera memory

			# Only the readout and nothing else will happen after this point.
			self.ready_to_prepare_next_sample.set()

			async with self._camera.readout():
				for i in range(num_frames):
					yield await self._camera.grab()


Imaging
-------

A basic frame acquisition generator which triggers the camera itself is provided by
:func:`.frames`

.. autofunction:: concert.experiments.imaging.frames

There are tomography helper functions which make it easier to define the proper
settings for conducting a tomographic experiment.

.. autofunction:: concert.experiments.imaging.tomo_angular_step

.. autofunction:: concert.experiments.imaging.tomo_projections_number

.. autofunction:: concert.experiments.imaging.tomo_max_speed



Synchrotron and X-Ray tube experiments
--------------------------------------

In :py:mod:`concert.experiments.synchrotron` and :py:mod:`concert.experiments.xraytube` implementations of Radiography, SteppedTomography,
ContinuousTomography and SteppedSpiralTomography, ContinuousSpiralTomography and GratingInterferometryStepping are implemented for the two different
source types.

For detailed information how they are implemented, you can have a look at the base classes :class:`concert.experiments.imaging.Radiography`,
:class:`concert.experiments.imaging.Tomography`, :class:`concert.experiments.imaging.SteppedTomography`, :class:`concert.experiments.imaging.ContinuousTomography`,
:class:`concert.experiments.imaging.SteppedSpiralTomography`, :class:`concert.experiments.imaging.ContinuousSpiralTomography` and :class:`concert.experiments.imaging.GratingInterferometryStepping`.

In the standard configuration, all tomography and radiography experiments first acquire the dark images, then the flat images and the projection images of the sample at the end.
This order can be adjusted by the :func:`~concert.experiments.base.Experiment.swap` command.


Radiography
"""""""""""

.. autoclass:: concert.experiments.imaging.RadiographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.RadiographyLogic
   :noindex:
.. autoclass:: concert.experiments.xraytube.RadiographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.LocalRadiography
   :noindex:
.. autoclass:: concert.experiments.synchrotron.RemoteRadiography
   :noindex:
.. autoclass:: concert.experiments.xraytube.LocalRadiography
   :noindex:
.. autoclass:: concert.experiments.xraytube.RemoteRadiography
   :noindex:


SteppedTomography
"""""""""""""""""

.. autoclass:: concert.experiments.imaging.SteppedTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.SteppedTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.xraytube.SteppedTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.LocalSteppedTomography
   :noindex:
.. autoclass:: concert.experiments.synchrotron.RemoteSteppedTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.LocalSteppedTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.RemoteSteppedTomography
   :noindex:


ContinuousTomography
""""""""""""""""""""

.. autoclass:: concert.experiments.imaging.ContinuousTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.ContinuousTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.xraytube.ContinuousTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.LocalContinuousTomography
   :noindex:
.. autoclass:: concert.experiments.synchrotron.RemoteContinuousTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.LocalContinuousTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.RemoteContinuousTomography
   :noindex:



SteppedSpiralTomography
"""""""""""""""""""""""

.. autoclass:: concert.experiments.imaging.SteppedSpiralTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.SteppedSpiralTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.xraytube.SteppedSpiralTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.LocalSteppedSpiralTomography
   :noindex:
.. autoclass:: concert.experiments.synchrotron.RemoteSteppedSpiralTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.LocalSteppedSpiralTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.RemoteSteppedSpiralTomography
   :noindex:


ContinuousSpiralTomography
""""""""""""""""""""""""""

.. autoclass:: concert.experiments.imaging.ContinuousSpiralTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.ContinuousSpiralTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.xraytube.ContinuousSpiralTomographyLogic
   :noindex:
.. autoclass:: concert.experiments.synchrotron.LocalContinuousSpiralTomography
   :noindex:
.. autoclass:: concert.experiments.synchrotron.RemoteContinuousSpiralTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.LocalContinuousSpiralTomography
   :noindex:
.. autoclass:: concert.experiments.xraytube.RemoteContinuousSpiralTomography
   :noindex:


GratingInterferometryStepping
"""""""""""""""""""""""""""""

In this grating based phase contrast imaging implementation a single projection is generated.
The grating is stepped with and without the sample while images are recorded.
Dark images are also recorded.
If the :class:`concert.experiments.addons.PhaseGratingSteppingFourierProcessing` addon is attached,
directly the intensity, visibility and differential phase are reconstructed.

.. autoclass:: concert.experiments.synchrotron.LocalGratingInterferometryStepping
   :noindex:
.. autoclass:: concert.experiments.xraytube.LocalGratingInterferometryStepping
   :noindex:


Control
-------

.. automodule:: concert.experiments.control
    :members:

Addons
------

Addons are special features which are attached to experiments and operate on
their data acquisition. For example, to save images on disk::

    from concert.experiments.addons import ImageWriter

    # Let's assume an experiment is already defined
    writer = ImageWriter(experiment.acquisitions, experiment.walker)
    writer.attach()
    # Now images are written on disk
    await experiment.run()
    # To remove the writing addon
    writer.detach()

.. automodule:: concert.experiments.addons
    :members:

Running an experiment
---------------------

To demonstrate how a typical experiment can be run in an empty session with dummy devices::

    from concert.storage import DirectoryWalker
    from concert.ext.viewers import PyplotImageViewer
    from concert.experiments.addons import Consumer, ImageWriter
    from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
    from concert.devices.camera.dummy import Camera
    from concert.devices.shutters.dummy import Shutter

    # Import experiment
    from concert.experiments.synchrotron import ContinuousTomography

    # Devices
    camera = await Camera()
    shutter = await Shutter()
    flat_motor = await LinearMotor()
    tomo_motor = await ContinuousRotationMotor()


    viewer = await PyplotImageViewer()
    walker = DirectoryWalker(root="folder to write data")
    exp = await ContinuousTomography(walker=walker,
                                     flat_motor=flat_motor,
                                     tomography_motor=tomo_motor,
                                     radio_position=0*q.mm,
                                     flat_position=10*q.mm,
                                     camera=camera,
                                     shutter=shutter)

    # Attach live_view to the experiment
    live_view = Consumer(exp.acquisitions, viewer)

    # Attach image writer to experiment
    writer = ImageWriter(exp.acquisitions, walker)

    # check all parameters by typing 'exp'

    # Run the experiment
    f = exp.run()

    # Wait until the experiment is done
    await f

