=========
Directors
=========

Directors can be employed to run :class:`concert.experiments.base.Experiment` multiple times.
The function :meth:`concert.directors.base._prepare_run` is used to prepare an experiment run.
E.g. this function can be used to exchange specimens or modify experiment properties.

The :py:attr:`.base.Experiment.ready_to_prepare_next_sample` can be used to trigger the :meth:`concert.directors.base._prepare_run` already while the experiment is still running.

.. autoclass:: concert.directors.base.Director
    :members:

XY Scanning
-----------

.. autoclass:: concert.directors.scanning.XYScan
    :members:


