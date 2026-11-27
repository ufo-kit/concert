.. _parameter-scanning:

===============
Parameter scans
===============

Scan experiments and scan directors visit combinations of device parameter
values. They share the same named axes, range editing, traversal orders and
dynamically installed Concert parameters.

Choose the appropriate runnable:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Action at each position
     - Data handling
   * - :class:`concert.experiments.scan.LocalScan`
     - Trigger and acquire one image
     - Yield images to local addons, with scan metadata
   * - :class:`concert.experiments.scan.RemoteScan`
     - Trigger and send one image
     - Send directly from the camera to remote addons
   * - :class:`concert.directors.scanning.Scan`
     - Run the supplied experiment
     - Use that experiment's local or remote acquisitions and addons

``concert.experiments.scan.Scan`` is a compatibility name for ``LocalScan``.
It is different from ``concert.directors.scanning.Scan``. Use import aliases
when working with both in the same session.


Acquire images locally
======================

This complete example uses dummy devices, irregular positions, a local image
writer and an accumulator::

    from concert.devices.cameras.dummy import Camera
    from concert.devices.motors.dummy import LinearMotor
    from concert.experiments.addons.local import Accumulator, ImageWriter
    from concert.experiments.scan import LocalScan
    from concert.quantities import q
    from concert.storage import DirectoryWalker

    camera = await Camera()
    x_motor = await LinearMotor()
    y_motor = await LinearMotor()
    walker = await DirectoryWalker(root='scan-data')

    images = await LocalScan(
        axes={
            'x': (x_motor['position'], [0, 2, 5] * q.mm),
            'y': (y_motor['position'], [0, 1] * q.mm),
        },
        camera=camera,
        order='snake',
        go_back=True,
        walker=walker,
    )
    writer = await ImageWriter(experiment=images)
    accumulator = await Accumulator(experiment=images)

    print(await images.get_num_positions())  # 6
    print(await images.get_scan_x())
    # position: values=[0 millimeter, 2 millimeter, 5 millimeter]; 3 points; order=snake

    await images.run()
    frames = await accumulator.get_items(images.scan)
    print(frames[0].metadata['scan'])

The scan has one acquisition, ``images.scan``. It awaits all parameter setters
before triggering the camera. A walker is optional for scan experiments; attach
an image writer when you want to store images. With the default
``separate_runs=True`` and ``name_fmt='run_{:>04}'``, each run uses its own run
directory. Images are written under the ``scan`` acquisition directory.


Run an experiment at every position
==================================

A director wraps an existing experiment. For example, to acquire two camera
exposures at every outer motor position::

    from concert.directors.scanning import Scan as ScanDirector

    exposures = await LocalScan(
        axes={'exposure': (camera['exposure_time'], [1, 5] * q.ms)},
        camera=camera, go_back=True, walker=walker,
    )
    exposure_writer = await ImageWriter(experiment=exposures)
    director = await ScanDirector(
        axes={
            'x': (x_motor['position'], [0, 2, 5] * q.mm),
            'y': (y_motor['position'], [0, 1] * q.mm),
        },
        experiment=exposures,
        order='snake',
        go_back=True,
    )

    print(await director.get_number_of_iterations())  # 6 experiments, 12 images
    await director.run()

The supplied experiment must have a walker. The director defaults to that
walker and uses ``director_run_{:>04}`` for its own run directory. It temporarily
sets the experiment's ``separate_runs`` to False and restores it afterwards.
Iteration directories describe the actual axis indices, for example
``iteration_0001_0000``, even when the traversal is snake or spiral.

.. warning::

   Directors support early preparation. When an experiment sets
   ``ready_to_prepare_next_outer_loop``, the director may move to the next
   position before the current experiment finishes. Signal readiness only when
   changing those parameters is safe. Without an early signal, the director
   waits until the experiment lifecycle finishes.

The director does not inject its outer axes into arbitrary experiment images.
When the supplied experiment is a ``LocalScan``, its image metadata describes
that experiment's own axes (``exposure`` in the example), not the director's
outer ``x`` and ``y`` axes.


Acquire images remotely
=======================

``RemoteScan`` uses the same axis configuration as ``LocalScan``. Given a
configured remote camera and walker::

    from concert.experiments.scan import RemoteScan

    remote_images = await RemoteScan(
        axes={
            'x': (x_motor['position'], [0, 2, 5] * q.mm),
            'y': (y_motor['position'], [0, 1] * q.mm),
        },
        camera=remote_camera,
        walker=remote_walker,
        order='snake', go_back=True,
    )
    # Attach configured remote addons to remote_images before running.
    await remote_images.run()

At each position it awaits the parameter setters, triggers the camera and
calls ``camera.grab_send(1, end=...)``. Only the final image ends the stream.
The camera must support remote acquisition endpoint registration and sending;
a dummy local camera is not a replacement for this setup. Remote addons must
be configured for the camera's endpoints. Local and remote consumers cannot be
mixed in the same acquisition. No images are returned to the local session,
and the session does not add local scan metadata to the remote images.


Axes, arbitrary lists and regular ranges
=======================================

The recommended constructor uses ``axes``, a dictionary mapping each axis name
to a ``(bound_parameter, values)`` pair. This keeps the name, device binding and
positions together rather than relying on matching entries in three parallel
lists. Dictionary insertion order defines the axis order, including traversal,
metadata indices and director directory indices::

    one_axis = await LocalScan(
        axes={'x': (x_motor['position'], [5 * q.mm, 0 * q.mm, 2 * q.mm])},
        camera=camera,
    )

The dictionary may also be supplied as the first positional argument::

    one_axis = await LocalScan(
        {'x': (x_motor['position'], [5, 0, 2] * q.mm)}, camera=camera)

Do not mix a mapping with ``parameters``, ``values`` or ``names``. Empty
mappings, invalid pairs, duplicate parameter bindings and conflicting generated
names are rejected. Axis names are taken directly from the dictionary keys.

The original constructor remains available for compatibility.
``parameters`` contains bound device parameters, such as ``motor['position']``;
``values`` contains one non-empty sequence of scalar positions per axis. A
one-dimensional scan accepts a single parameter and a single value sequence::

    one_axis = await LocalScan(
        x_motor['position'], [5 * q.mm, 0 * q.mm, 2 * q.mm],
        camera=camera, names=['x'],
    )

Lists may be unsorted, irregularly spaced or contain repeated values. They are
never sorted or deduplicated. Repeated values at distinct indices result in
distinct acquisitions or experiment runs. Quantities in a list may have
different compatible units; the bound device parameter performs unit conversion
and enforces its limits when the scan runs. Construction validates the list
shape, not every future hardware setting.

Names must be unique non-empty strings. Defaults are ``axis_0``, ``axis_1``,
etc. ``scan.axes`` is a read-only mapping preserving the supplied axis order.
Axis bindings cannot be replaced, but their configured positions can be edited::

    await images.axes['x'].set_values([8 * q.mm, 1 * q.mm, 4 * q.mm])
    await images.axes['y'].set_range(0 * q.mm, 3 * q.mm, 2 * q.mm)
    # y positions: 0 mm, 2 mm, 3 mm

    await images.axes['y'].set_range(
        0 * q.mm, 3 * q.mm, 2 * q.mm, include_last=False)
    # y positions: 0 mm, 2 mm

    await images.axes['y'].set_range(3 * q.mm, 0 * q.mm, -2 * q.mm)
    # y positions: 3 mm, 1 mm, 0 mm

``set_range`` includes the exact stop value by default, even if the step does
not divide the range evenly. Thus the last interval can be shorter. Negative
steps are required for descending ranges. Zero steps, steps pointing away from
the stop, non-finite range arguments and empty results are rejected. An invalid
replacement leaves the previous positions intact.

``set_values`` switches an axis to explicit-list mode and clears its recorded
start, stop and step. ``set_range`` replaces all three range settings and the
generated positions atomically. Range fields are read-only; configure ranges
with ``set_range`` rather than changing endpoints individually.


Traversal orders
================

Set ``order`` in either constructor, or change it between runs::

    await images.set_order('spiral')
    print(await images.get_position_indices(3))  # (2, 1) for a 3-by-2 grid

.. list-table::
   :header-rows: 1

   * - Order
     - Dimensions
     - Behavior
   * - ``raster`` (default)
     - One or more
     - Cartesian traversal; last axis fastest
   * - ``snake``
     - One or more
     - Reflected traversal; successive points have adjacent grid indices
   * - ``spiral``
     - Exactly two
     - Clockwise outside-in, starting at ``(0, 0)`` along the second axis

These grids show the zero-based acquisition iteration at each position of a
3-by-3 scan. Rows are the first axis and columns are the second axis::

    raster         snake          spiral
    0  1  2        0  1  2        0  1  2
    3  4  5        5  4  3        7  8  3
    6  7  8        6  7  8        6  5  4

Spiral also supports rectangular, single-row and single-column grids. It does
not start at the center and is not a continuous geometric spiral. Snake in
higher dimensions reflects the nested traversal so row and plane boundaries
remain adjacent in index space.

All orders operate on list indices, not physical distances. Unsorted lists can
still cause large device moves. ``get_position_indices(iteration)`` returns a
tuple in axis order and raises ``IndexError`` outside the current grid. The
total number of positions is the product of the axis lengths, independent of
the order. The process-level ``concert.processes.common.scan`` also accepts
the same ``order`` keyword.


Inspect and edit Concert parameters
==================================

The normal ``info_table`` and recorded runnable metadata include the following
shared Concert parameters:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Writable
     - Meaning
   * - ``scan_plan``
     - No
     - Table string containing the order, axis names, parameter names, ranges,
       point counts and all configured values
   * - ``num_positions``
     - No
     - Total number of Cartesian positions
   * - ``order``
     - Yes, between runs
     - ``raster``, ``snake`` or ``spiral``
   * - ``go_back``
     - Yes, between runs
     - Restore scanned device parameters after the scan

Directors additionally expose ``number_of_iterations``, equal to
``num_positions``. Both experiments and directors automatically install the
following parameters for each axis named ``x``:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Writable
     - Meaning
   * - ``scan_x``
     - No
     - Readable live summary of binding, values/range, count and order
   * - ``scan_x_values``
     - Yes, between runs
     - Actual configured positions (returned as a tuple)
   * - ``scan_x_start``, ``scan_x_stop``, ``scan_x_step``
     - No
     - Recorded range settings, or None for explicit positions
   * - ``scan_x_num_points``
     - No
     - Number of positions on this axis

These parameters are live views, not separate copies of the configuration::

    print(await images.info_table)
    print(await images.get_scan_plan())
    print(await images.get_scan_x())
    await images.set_scan_x_values([8, 1, 4] * q.mm)
    # Equivalent:
    await images['scan_x_values'].set([8, 1, 4] * q.mm)
    print(await images.axes['x'].get_values())

In a synchronous Concert shell, normal parameter-property access such as
``images.scan_x`` and ``images.scan_x_values = positions`` is also available.
Use the awaitable getters/setters inside asynchronous functions.

Dynamic parameters are specific to each instance. Names are prefixed with
``scan_`` and characters other than ASCII letters, digits or underscores are
replaced with underscores. For example ``sample x`` becomes ``scan_sample_x``.
The mapping is available programmatically::

    summary_name = images.axis_parameter_names['x']['summary']
    values_name = images.axis_parameter_names['x']['values']
    print(await images[summary_name].get())
    await images[values_name].set([1, 4] * q.mm)

Generated names must not conflict with existing attributes or another axis's
generated parameters. For example ``x`` and ``x_values`` conflict; ``sample x``
and ``sample-x`` also conflict after sanitization. Construction rejects these
names rather than overwriting parameters.

The plan string currently includes the full tuple of values using their Python
representation. Large lists can make it wide; use ``get_scan_x()`` for an
individual readable summary or ``get_scan_x_values()`` for programmatic access.


.. _scan-image-metadata:

Local image metadata
====================

``LocalScan`` yields :class:`concert.helpers.ImageWithMetadata` objects. Each
frame contains this JSON-serializable ``metadata['scan']`` schema::

    {
        'iteration': 3,
        'indices': [1, 0],
        'order': 'snake',
        'axes': {
            'x': {'parameter': 'position', 'value': '2 millimeter'},
            'y': {'parameter': 'position', 'value': '0 millimeter'},
        },
    }

``iteration`` is the acquisition number within this scan run, starting at zero.
``indices`` contains the visited index in each axis's configured value list.
``axes`` is keyed by the original axis names, distinguishing parameters such
as two motors both named ``position``. Each entry stores the parameter name and
its actual read-back value as a string including units. The value may differ
from the requested value, and may use the device's units rather than those in
the original list.

Read-backs are sampled sequentially immediately before triggering, after the
scan's setters have completed; this is not an atomic multi-device snapshot.
Only scanned parameters are included, not all camera or experiment parameters.
Each frame has an independent metadata snapshot, so later moves, configuration
edits or restoration do not change previous frames. Existing camera metadata,
including orientation and conversion flags, is preserved; the ``scan`` key is
reserved for the new scan snapshot and replaces an existing entry of that name.

Local image writers receive this metadata with each frame and can serialize it
using their normal metadata handling. For example, read a frame's values from
an accumulator as follows::

    frames = await accumulator.get_items(images.scan)
    position = frames[0].metadata['scan']['axes']['x']['value']
    print(position)  # '0 millimeter' for the original example

Remote scans do not add this metadata in the local session. Scan directors do
not automatically add their axes to the supplied experiment's images.


Lifecycle and restrictions
==========================

* Every run starts at the first position in the selected traversal. Completing
  a run does not resume the next run at the previous endpoint.
* Positions, order and ``go_back`` may be changed in standby, error or cancelled
  states, but not while running or paused. This also applies to generated
  Concert setters. Changing configuration alone does not move devices.
* ``go_back=False`` is the default and leaves parameters at their last visited
  values. ``go_back=True`` stashes and restores the scanned parameters when the
  scan completes, fails or is cancelled. Restoration still requires working
  device setters; cleanup cannot recover unavailable or failing hardware.
* Image scans stop an already-recording camera, use software triggering and
  restore its previous trigger source after acquisition. They stop recording
  afterwards; they do not resume a previous recording session.
* A parameter must be bound and writable, and cannot appear on multiple axes.
  Image scans control the camera trigger source themselves, so that parameter
  cannot be a scan axis. Device-specific limitations on changing settings while
  recording still apply.
* Image scans do not open/close shutters or acquire dark/flat fields. Supply
  appropriate preparation/cleanup or use a director with a suitable imaging
  experiment when such acquisition steps are required.
* Directors retain the existing pause/resume and cancellation semantics. Pause
  takes effect before starting the next experiment, after any early preparation;
  cancellation is treated by the base director as a normal stop.

See :doc:`../../devel/reference/scanning` for the shared API reference.
