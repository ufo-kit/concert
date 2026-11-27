============
Installation
============

openSUSE packages
=================

We use the `openSUSE Build Service`__ to provide packages for openSUSE 12.2
until openSUSE 13.2. Add the repository first, e.g.::

    $ sudo zypper ar http://download.opensuse.org/repositories/home:/ufo-kit/openSUSE_12.2/ concert-repo

and update and install the packages. Note, that you have to install IPython on
your own, if you intend to use the ``concert`` binary for execution::

    $ sudo zypper update
    $ sudo zypper in python-concert

__ https://build.opensuse.org/project/show/home:ufo-kit


Install from PyPI
=================

It is recommended to use pip_ for installing Concert. The fastest way to install
it is from PyPI::

    $ sudo pip install concert

This will install the latest stable version. If you prefer an earlier stable
version, you can `fetch a tarball`_ and install with::

    $ sudo pip install concert-x.y.z.tar.gz

If you haven't have pip_ available, you can extract the tarball and install using
the supplied ``setup.py`` script::

    $ tar xfz concert-x.y.z.tar.gz
    $ cd concert-x.y.z
    $ sudo python setup.py install

More information on installing Concert using the ``setup.py`` script, can be
found in the official `Python documentation`__.

To install the Concert from the current source, follow the instructions given in
the :ref:`developer documentation <get-the-code>`.

__ http://docs.python.org/2/install/index.html


GUI dependencies
----------------

The parameter widgets and standalone GUI use PyQt6 (6.12.1 or newer),
pyqtgraph, and qasync. Install these optional dependencies from the source
checkout with::

    $ python -m pip install -e '.[gui]'

Use PyQt6 consistently in applications embedding these widgets; do not mix
Qt bindings in the same process. If you explicitly configure binding selection,
set ``PYQTGRAPH_QT_LIB=PyQt6`` and ``QT_API=pyqt6`` before starting Python.

Interactive sessions remain headless by default. Enable GUI event processing
explicitly with ``concert start --with-gui``. Importing ``concert.gui`` does not
create a Qt application; standalone applications should call
``concert.gui.get_application()`` before constructing any widgets.

Parameter widget constructors do not block on devices. Each parameter is read
once automatically when it first becomes visible and the asyncio loop is running.
Explicitly refresh values with ``await widget.refresh()`` and enable periodic reads with
``widget.set_polling(True)``. Call ``await widget.shutdown()`` before destroying
the containing window or stopping the event loop. Supported generic editors are
strings, integers, floats, and booleans; other generic values are read-only.
Quantity and selection parameters have dedicated editors.
Automatic reads, polling, and ``widget.refresh()`` only access rows visible in
the current viewport. Switching tabs, minimizing the window, or scrolling rows
out of view pauses their polling without clearing the Poll preference. It resumes
when they return. Use ``await widget.params[name].read()`` for a deliberate read
regardless of visibility.

Refresh configuration and layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each panel uses one shared visibility/refresh clock. Standalone rows have a
lightweight scheduler of their own. Poll durations are positive Concert time
quantities (not unitless milliseconds)::

    widget = ParameterizableWidget(
        motor, poll_interval=0.5 * q.s, initial_read=True,
        refresh_when_visible=True,
        parameter_options={'position': {'poll_interval': 1 * q.s}},
        groups={'Motion': ['position']}, order=['state'])

``initial_read=False`` disables initialization reads. ``refresh_when_visible``
controls refreshing an already-read value when it reappears; enabled polling
always resumes on visibility. ``parameter_options`` can override all three
settings per row. Group headings are checkable: unchecking collapses their rows
and pauses updates. Unlisted parameters remain available after grouped/ordered
parameters. Unknown names and duplicate group membership are rejected.

Parameters expose public ``help``, ``readable``, ``writable`` and
``target_readable`` metadata without device I/O; quantities also expose ``unit``
and ``limits_locked``. Write-only parameters are never automatically read.

Editor overrides
~~~~~~~~~~~~~~~~

An editor adapter is a small object with a ``widget``, an ``editable`` property,
a ``display(value)`` method, and a ``parse()`` method. It knows how to show and
convert values, but never talks to the device. The controller handles device
access, validation, and operation ownership independently of Qt.

Most applications need no configuration: built-in adapters handle scalar,
quantity, and selection parameters. To customize one parameter, pass an editor
factory by name::

    widget = ParameterizableWidget(camera, editors={'exposure_time': ExposureEditor})

``ExposureEditor(param)`` must return an adapter implementing that interface.
This also permits explicit editors for writable parameters whose value cannot
be read. For reuse across an application, create an ``EditorRegistry``, call
``registry.register(ParameterValueSubclass, EditorFactory)``, and pass it as
``registry=registry``. Recently registered factories take precedence over the
built-in defaults. No changes to Concert's device classes are required.
Built-in text, checkbox, and selection editors automatically track edits.
Custom controls should connect their user-change signal to ``row.mark_dirty``.

Quantity editors show numeric values and units separately and accept either a
number in the displayed unit or explicit compatible units. Writes check current
soft limits before device submission. These checks do not replace hardware
interlocks or the validation performed by device setters. Actual values and
commanded targets are displayed separately, with help, limits, busy state, and
error information available in the row and its tooltips. Effective, user, and
external limits and their lock state are in a collapsed ``Limits`` section.
Expand it to inspect the bounds or edit user lower/upper limits. Each bound has
its own Apply button and uses Concert's existing async limit setter; empty input
removes that user bound. External limits remain read-only and enforced. Locked
limits disable editing. Changes do not move the device or reset its read age.
Unsaved limit edits survive refreshes; Reset limit edits restores the last read
bounds. Absent bounds are marked Unbounded and failed metadata Unavailable.

Unsaved input is marked Edited and never overwritten by refreshes. Observed
values and targets continue updating independently. Reset restores the latest
observed value, and a successful Apply clears the edit marker. Polling does not
prevent editing; pending writes remain serialized. Optional metadata failures
are separate from actual-value failures and do not invalidate a successful read.

Applications can connect ``widget.operation_failed`` to a handler accepting
``(parameter_name, operation, exception)``. Operations are ``read``, ``write``,
``limits`` (user-limit changes), ``validation`` (input parsing), or ``metadata``
(optional target/limits reads).

Lifecycle and cancellation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hiding or closing a widget stops polling and cancels reads, but does **not**
cancel a submitted write. ``shutdown()`` waits for writes to finish. Closing the
standalone window similarly waits for an active experiment rather than treating
window closure as a hardware abort. Explicit hardware abort remains a separate
Concert operation; task cancellation cannot guarantee a motor stopped or a
write had no effect.

``GuiSession`` owns panels/controllers and event integration. Standalone qasync apps
use ``GuiSession(pump=False)``; IPython sessions use ``pump=True``. Register panels
with ``ParameterizableWidget(device, session=session)`` and await
``session.shutdown()`` before closing the event loop. The GUI event pump is not
a Concert hardware task and survives the normal background-abort shortcut.
Panel shutdown stops its scheduler, waits for writes and detaches it from the
session. ``session.register(resource)`` also returns a callback for explicitly
detaching a resource after shutdown.

Interactive alignment
~~~~~~~~~~~~~~~~~~~~~

Alignment is an asynchronous workflow object, not a parameterizable device.
Alignment marks the desired center and selected point on the image. A left
click proposes a displacement and opens a confirmation prompt (default: No).
Right-click, closing the viewer, or task cancellation stops the workflow. All
translation-axis soft limits are checked before any translation is submitted.
For automated applications, supply ``confirmation=callable``; it receives the
two displacement quantities and returns whether to proceed.
The callable may also be asynchronous; the built-in confirmation dialog uses
nonblocking Qt dispatch so device tasks remain responsive while it is open.

The rotation motor remains at the acquired alignment angle on cancellation.
Camera configuration is restored, but motor moves are **not** rolled back. If
a movement fails partway through, the error reports the completed axes and the
operator must inspect actual positions before retrying. These are UI safeguards,
not a substitute for beamline safety controls.

To run the GUI compatibility tests without a display::

    $ python -m pip install pytest-qt
    $ QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyqt6 python -m pytest concert/tests/unit/gui concert/tests/unit/test_gui_pyqt6.py concert/tests/unit/test_gui_startup.py concert/tests/unit/test_gui_controller.py

Controller and startup tests run without Qt. Normal widget tests use pytest-qt
and a controlled qasync loop; process-global startup and workflow smoke tests
run in isolated subprocesses.


Installing into a virtualenv
----------------------------

It is sometimes a good idea to install third-party Python modules independent of
the system installation. This can be achieved easily using pip_ and virtualenv_.
When virtualenv is installed, create a new empty environment and activate that
with ::

    $ virtualenv my_new_environment
    $ . my_new_environment/bin/activate

Now, you can install Concert's requirements and Concert itself ::

    $ pip install -e path_to_concert/

As long as ``my_new_environment`` is active, you can use Concert.


.. _pip: https://pypi.python.org/pypi
.. _`fetch a tarball`: https://github.com/ufo-kit/concert/releases
.. _virtualenv: http://virtualenv.org
