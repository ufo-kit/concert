========
Sessions
========

Each beamline consists of many devices, of which only a subset is useful in a
particular experiment. To group this subset in a meaningful way, Concert
provides a session mechanism managed by the ``concert`` command line tool.


Three-minute tour
=================

The ``concert`` tool is run from the command line.  Without any arguments, its
help is shown::

    $ concert
    usage: concert [-h] [--version]  ...

    optional arguments:
      -h, --help  show this help message and exit
      --version   show program's version number and exit

    Concert commands:

        log       Show session logs
        show      Show available sessions or details of a given *session*
        edit      Edit a session
        start     Start a session
        init      Create a new session
        rm        Remove one or more sessions

The tool is command-driven, that means you call it with a command as its first
argument. To read command-specific help, use::

    $ concert [command] -h

Now, let's get started and create a new session. For this, we use the ``init``
command with a name for our new session::

    concert init experiment

This creates a new session *experiment*. If such a session already exists,
Concert will warn you. You can overwrite the existing session with ::

    concert init --force experiment


.. note::

    The location of the session files depends on the chosen installation method.
    If you installed into a virtual environment ``venv``, the files will be
    stored in ``/path/to/venv/share/concert``. If you have installed Concert
    system-wide our without using a virtual environment, it is installed into
    ``$XDG_DATA_HOME/concert`` or ``$HOME/.local/share/concert`` if the former
    is not set. See the `XDG Base Directory Specification
    <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_
    for further information. It is probably a *very* good idea to put the
    session directory under version control.

A session is already populated with useful imports, but you will most likely
add more definitions. For this ::

    concert edit experiment

will load the experiment session in an editor for you. Once, the session is
saved, you can start it by ::

    concert start experiment

This will load an IPython shell and import all definitions from the session
file. To remove a session, you can use the ``rm`` command::

    concert rm experiment

During an experiment, devices will output logging information. By default, this
information is gathered in a central file. To view the log for all experiments
use ::

    concert log

and to view the log for a specific experiment use ::

    concert log experiment


.. note::

    When Concert is installed system-wide, a bash completion for the
    ``concert`` tool is installed too. This means, that commands and options
    will be completed when pressing the :kbd:`Tab` key.


Writing a Concert session
=========================

Concert uses the quantities_ package to represent units in a programmatical way.
Therefore, you will see the quantities module *imported* as the name ``q`` into
your session. When you start the session you can use it right away to do unit
calculation::

    >>> a = 9.81 * q.m / q.s**2
    >>> v = 5 * q.s * a
    >>> "Velocity after 5 seconds: {0}".format(v)
    'Velocity after 5 seconds: 49.05 m/s'

You will also notice the placeholder text assigned to the ``__doc__`` variable.
This should be change to something descriptive as it will be shown each time you
start the session.

Adding devices
--------------

To create a device suited for your experiment you have to import it first.
Concert uses the following packaging scheme to separate device classes and
device implementations: ``concert.devices.[class].[implementation]``. Thus if
you want to create a dummy ring from the storage ring class, you would add this
line to your session::

    from concert.devices.storagerings.dummy import DummyRing

Once imported, you can create the device and give it a name that will be
accessible from the command line shell::

    from concert.devices.motors.base import LinearCalibration
    from concert.devices.motors.dummy import DummyMotor

    ring = DummyRing()

    # Create a motor that moves one step per millimeter without an offset
    calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
    motor = DummyMotor(calibration)

To access a device, you can use the dot notation to read and write its parameters::

    >>> motor.position = 2 * q.mm

For more information on how to *use* devices, see :ref:`controlling-devices`.

.. note::

   If a device requires a unit for one of its parameters, you *must* use it.
   This ensures consistent results throughout an experiment. However, you are
   free to use any prefixed unit, like millimeter, centimeter or kilometer for a
   motor's position.


Show information
----------------

To get information about the state of a device, you can simply print it::

    >>> print(ring)
     Parameter  Value
     current    99.45 mA
     energy     4.45 MeV
     lifetime   11.0 h

This can become tiresome if you have several devices. To fix this, we can use a
session's ``ddoc`` and ``dstate`` list. All you have to do is adding the desired
devices to these lists::

    from concert.session import ddoc, dstate

    devices = [motor, ring]
    ddoc.extend(devices)
    dstate.extend(devices)

Now, you simple get the state and information about all devices via ``dstate``
and ``ddoc`` ::

    >>> dstate
    ---------------------------------------------
      Name         Parameters
    ---------------------------------------------
      DummyMotor    position  99.3820097256 mm
                    state     standby
    ---------------------------------------------
      DummyRing     current   99.45 mA
                    lifetime  11.0 h
                    energy    4.45 MeV
    ---------------------------------------------

    >>> ddoc
    ------------------------------------------------------------------------------
      Name         Description   Parameters
    ------------------------------------------------------------------------------
      DummyMotor   None           Name      Access  Unit  Description
                                  position  rw      m     Position of the motor
                                  state     r       None  None
    ------------------------------------------------------------------------------
      DummyRing    None           Name      Access  Unit  Description
                                  current   r       mA    Current of the ring
                                  lifetime  r       h     Lifetime of the ring
                                  energy    r       MeV   Energy of the ring
    ------------------------------------------------------------------------------


.. _quantities: https://pypi.python.org/pypi/quantities


Importing other sessions
------------------------

To specify experiments that share a common set of devices, you can define a base
session and import it from each sub-session::

    from base import *

Now everything that was defined will be present when you start up the new
session.


Customizing log output
======================

By default, logs are gathered in ``$XDG_DATA_HOME/concert/concert.log``. To
change this, you can pass the ``--logto`` and ``--logfile`` options to the
``start`` command. For example, if you want to output log to ``stderr`` use ::

    concert --logto=stderr start experiment

or if you want to get rid of any log data use ::

    concert --logto=file --logfile=/dev/null start experiment
