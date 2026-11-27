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

Parameter scanning
------------------

See :ref:`parameter-scanning` for complete examples and the shared axis,
traversal-order and dynamic-parameter reference.

:class:`concert.directors.scanning.Scan` sets one or multiple device parameters
and runs an experiment at each Cartesian position. Its named axes and range
editing are shared with :class:`concert.experiments.scan.LocalScan` and
:class:`concert.experiments.scan.RemoteScan`::

    from concert.directors.scanning import Scan
    from concert.quantities import q

    director = await Scan(
        axes={
            'x': (x_motor['position'], [0, 1, 2] * q.mm),
            'y': (y_motor['position'], [0, 5] * q.mm),
        },
        experiment=experiment,
        go_back=True,
    )

    print(await director.get_scan_plan())
    print(await director.get_number_of_iterations())  # 6
    await director.axes['x'].set_range(0 * q.mm, 10 * q.mm, 2 * q.mm)
    await director.axes['y'].set_values([0, 1, 5] * q.mm)
    await director.run()

For a single axis, pass a single bound parameter and its positions. Values may
be arbitrary lists, including irregular spacing, unsorted or repeated values::

    director = await Scan(
        [x_motor['position'], y_motor['position']],
        [[5 * q.mm, 0 * q.mm, 2 * q.mm], [1 * q.mm, 7 * q.mm, 3 * q.mm]],
        experiment=experiment, names=['x', 'y'], order='snake',
    )
    await director.axes['x'].set_values([8 * q.mm, 1 * q.mm, 4 * q.mm])
    await director.set_order('spiral')
    print(await director.get_position_indices(3))

``order='raster'`` is the default, with the last axis changing fastest.
``snake`` reverses alternating rows and supports any number of axes.
``spiral`` requires exactly two axes and scans clockwise from the outside in,
starting at index ``(0, 0)`` along the second axis. Orders follow list indices,
not physical distances or sorted values. Each combination is visited once,
including repeated values at distinct list indices. Order can be adjusted only
between runs and is shown by ``get_scan_plan``.

Iteration directories encode the actual visited axis indices, for
example ``iteration_0001_0002``. The supplied experiment can be local or remote
and must have a walker, like experiments used by other directors.

Each axis also gets dynamically installed Concert parameters directly on the
director: ``scan_x`` (live readable summary), ``scan_x_values`` (editable
positions), and read-only ``scan_x_start``, ``scan_x_stop``, ``scan_x_step`` and
``scan_x_num_points``. They appear in the director's normal ``info_table`` and
metadata, and stay synchronized with ``director.axes['x']``::

    print(await director.info_table)
    print(await director.get_scan_x())
    await director.set_scan_x_values([8, 1, 4] * q.mm)
    await director.axes['x'].set_range(0 * q.mm, 10 * q.mm, 2 * q.mm)

``director.axis_parameter_names`` provides the generated names for all axes.

The director temporarily disables the experiment's ``separate_runs`` setting
and restores it afterwards. With ``go_back=True``, scanned parameters are also
restored when the scan finishes or stops. Axes cannot be edited while running
or paused. Each director run starts scanning from the first position again.

The existing early-preparation behavior is preserved: if the experiment sets
``ready_to_prepare_next_outer_loop``, the next position may be set before the
current experiment finishes. Only signal readiness when changing the scanned
parameters is safe for the current experiment.

.. autoclass:: concert.directors.scanning.Scan
    :members:
    :noindex:


XY Scanning
-----------

.. autoclass:: concert.directors.scanning.XYScan
    :members:
