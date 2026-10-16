=========
Directors
=========

Directors can be employed to run :class:`concert.experiments.base.Experiment` multiple times.
The function :meth:`concert.directors.base.Director._prepare_run` is used to prepare an experiment
run. E.g. this function can be used to exchange specimens or modify experiment properties.
Implement :meth:`concert.directors.base.Director.has_next_iteration` to decide at runtime whether
another iteration is needed. For a count known before the run, inherit from
:class:`concert.directors.base.FixedIterationDirector` and implement ``_get_number_of_iterations``.

For example, a director may stop as soon as an analysis object reports that enough data has been
collected::

    class ResultDrivenDirector(Director):
        async def has_next_iteration(self, iteration):
            return not analysis.is_complete

        async def _prepare_run(self, iteration):
            await sample_motor.set_position(positions[iteration])

``has_next_iteration`` is called when the experiment sets
``ready_to_prepare_next_outer_loop``. The experiment should therefore update the result used by the
stopping condition before setting that event.

The :py:attr:`.base.Experiment.ready_to_prepare_next_sample` can be used to trigger
:meth:`concert.directors.base.Director._prepare_run` already while the experiment is still running.

.. autoclass:: concert.directors.base.Director
    :members:

.. autoclass:: concert.directors.base.FixedIterationDirector
    :members:

XY Scanning
-----------

.. autoclass:: concert.directors.scanning.XYScan
    :members:
