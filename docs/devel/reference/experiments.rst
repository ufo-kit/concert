Experiments
===========


There are abstract implementations for radiography, stepped tomography, continuous tomography,
stepped spiral tomography and continuous spiral tomography.

All of them implement :class:`.LocalAcquisition` for dark images (without beam), flat field images (with beam, but sample moved to
Experiment.flatfield_position) and projections of the sample according to the measurement scheme.

In each acquisition generator the functions :func:`~concert.experiments.imaging.Radiography._prepare_flats`, :func:`~concert.experiments.imaging.Radiography._finish_flats`,
:func:`~concert.experiments.imaging.Radiography._prepare_darks`, :func:`~concert.experiments.imaging.Radiography._finish_darks`,
:func:`~concert.experiments.imaging.Radiography._prepare_radios`, :func:`~concert.experiments.imaging.Radiography._finish_radios` are called.
Overwriting them allows an easy way to implement special features within the experiments.

To use the classes one has to implement the :func:`~concert.experiments.imaging.Radiography.start_sample_exposure` and
:func:`~concert.experiments.imaging.Radiography.stop_sample_exposure` accordingly
(see :class:`concert.experiments.synchrotron.SynchrotronMixin` as an example).

For special cameras the generator :func:`~concert.experiments.imaging.Radiography._produce_frames` can be overwritten.


Radiography
"""""""""""

.. autoclass:: concert.experiments.imaging.RadiographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.RadiographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.RadiographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.LocalRadiography
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.RemoteRadiography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.LocalRadiography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.RemoteRadiography
   :show-inheritance:


SteppedTomography
"""""""""""""""""

.. autoclass:: concert.experiments.imaging.SteppedTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.SteppedTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.SteppedTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.LocalSteppedTomography
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.RemoteSteppedTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.LocalSteppedTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.RemoteSteppedTomography
   :show-inheritance:


ContinuousTomography
""""""""""""""""""""

.. autoclass:: concert.experiments.imaging.ContinuousTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.ContinuousTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.ContinuousTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.LocalContinuousTomography
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.RemoteContinuousTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.LocalContinuousTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.RemoteContinuousTomography
   :show-inheritance:


SteppedSpiralTomography
"""""""""""""""""""""""

.. autoclass:: concert.experiments.imaging.SteppedSpiralTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.SteppedSpiralTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.SteppedSpiralTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.LocalSteppedSpiralTomography
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.RemoteSteppedSpiralTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.LocalSteppedSpiralTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.RemoteSteppedSpiralTomography
   :show-inheritance:


ContinuousSpiralTomography
""""""""""""""""""""""""""

.. autoclass:: concert.experiments.imaging.ContinuousSpiralTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.ContinuousSpiralTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.ContinuousSpiralTomographyLogic
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.LocalContinuousSpiralTomography
   :show-inheritance:
.. autoclass:: concert.experiments.synchrotron.RemoteContinuousSpiralTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.LocalContinuousSpiralTomography
   :show-inheritance:
.. autoclass:: concert.experiments.xraytube.RemoteContinuousSpiralTomography
   :show-inheritance:


GratingInterferometryStepping
"""""""""""""""""""""""""""""

In this grating based phase contrast imaging implementation a single projection is generated.
The grating is stepped with and without the sample while images are recorded.
Dark images are also recorded.
If the :class:`concert.experiments.addons.PhaseGratingSteppingFourierProcessing` addon is attached,
directly the intensity, visibility and differential phase are reconstructed.

.. autoclass:: concert.experiments.synchrotron.LocalGratingInterferometryStepping
    :show-inheritance:
.. autoclass:: concert.experiments.xraytube.LocalGratingInterferometryStepping
    :show-inheritance:
