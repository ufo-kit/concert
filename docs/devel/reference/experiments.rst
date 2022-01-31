Experiments
===========


There are abstract implementations for radiography, stepped tomography, continuous tomography,
stepped spiral tomography and continuous spiral tomography.

All of them implement :class:`.Acquisition` for dark images (without beam), flat field images (with beam, but sample moved to
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
-----------

.. autoclass:: concert.experiments.imaging.Radiography
    :show-inheritance:
    :members: start_sample_exposure, stop_sample_exposure, _produce_frames, _prepare_darks, _finish_darks, _prepare_flats, _finish_flats,  _prepare_radios, _finish_radios, _take_radios, _take_darks, _take_flats



Tomography
----------

.. autoclass:: concert.experiments.imaging.Tomography
    :show-inheritance:
    :members: flat_position, radio_position, num_darks, num_flats, num_projections, angular_range, start_angle, start_sample_exposure, stop_sample_exposure, _produce_frames, _prepare_darks, _finish_darks, _prepare_flats, _finish_flats,  _prepare_radios, _finish_radios, _take_radios, _take_darks, _take_flats

Stepped tomography
------------------

.. autoclass:: concert.experiments.imaging.SteppedTomography
    :show-inheritance:
    :members: flat_position, radio_position, num_darks, num_flats, num_projections, angular_range, start_angle, start_sample_exposure, stop_sample_exposure, _produce_frames, _prepare_darks, _finish_darks, _prepare_flats, _finish_flats,  _prepare_radios, _finish_radios, _take_radios, _take_darks, _take_flats

Continuous tomography
---------------------

.. autoclass:: concert.experiments.imaging.ContinuousTomography
    :show-inheritance:
    :members: flat_position, radio_position, num_darks, num_flats, num_projections, angular_range, start_angle, velocity, start_sample_exposure, stop_sample_exposure, _produce_frames, _prepare_darks, _finish_darks, _prepare_flats, _finish_flats,  _prepare_radios, _finish_radios, _take_radios, _take_darks, _take_flats

Stepped spiral tomography
-------------------------

.. autoclass:: concert.experiments.imaging.SteppedSpiralTomography
    :show-inheritance:
    :members: flat_position, radio_position, num_darks, num_flats, num_projections, angular_range, start_angle, start_position_vertical, vertical_shift_per_tomogram, sample_height, start_sample_exposure, stop_sample_exposure, _produce_frames, _prepare_darks, _finish_darks, _prepare_flats, _finish_flats,  _prepare_radios, _finish_radios, _take_radios, _take_darks, _take_flats

Continuous spiral tomography
----------------------------

.. autoclass:: concert.experiments.imaging.ContinuousSpiralTomography
    :show-inheritance:
    :members: flat_position, radio_position, num_darks, num_flats, num_projections, angular_range, start_angle, velocity,  start_position_vertical, vertical_shift_per_tomogram, sample_height, start_sample_exposure, stop_sample_exposure, _produce_frames, _prepare_darks, _finish_darks, _prepare_flats, _finish_flats,  _prepare_radios, _finish_radios, _take_radios, _take_darks, _take_flats
