========
Tutorial
========

Concert is primarily a user interface to control devices commonly found at a
Synchrotron beamline. This guide will briefly show you how to *use* it and how
to *extend* it


Running a session
=================

In case you don't have a beamline at hand, you can fetch our sample sessions
with the :ref:`fetch <fetch-command>` command::

    $ concert fetch --repo https://github.com/ufo-kit/concert-examples

Now :ref:`start <start-command>` the tutorial session::

    $ concert start tutorial

Concert uses the pint_ package to represent units in a programmatical way.
Therefore, you will see ``UnitRegistry`` instance *imported* as the name ``q``
into your session. When you start the session you can use it right away to do
unit calculation::

    tutorial > a = 9.81 * q.m / q.s**2
    tutorial > v = 5 * q.s * a
    tutorial > "Velocity after 5 seconds: {0}".format(v)

    'Velocity after 5 seconds: 49.05 meter / second'

You will be greeted by an IPython shell loaded with pre-defined devices and
processes. You can get an overview of all defined devices by calling the
:func:`~.dstate` and :func:`~.ddoc` functions::

    tutorial > dstate()

    ---------------------------------------------
      Name         Parameters
    ---------------------------------------------
      DummyMotor    position  99.382 millimeter
                    state     standby
    ---------------------------------------------
      ... 

    tutorial > ddoc()

    ------------------------------------------------------------------------------
      Name         Description   Parameters
    ------------------------------------------------------------------------------
      DummyMotor   None           Name      Access  Unit  Description
                                  position  rw      m     Position of the motor
                                  state     r       None  None
    ------------------------------------------------------------------------------
      ...

If you just type the name of a device, it will print out the current parameter
values of it::

    tutorial > motor

    <concert.devices.motors.dummy.Motor object at 0x9419f0c>
     Parameter  Value                    
     position   12.729455653 millimeter

:func:`.pdoc` displays information about currently defined functions and
processes and may look like this::

    tutorial > pdoc()
    ------------------------------------------------------------------------------
    Name                   Description
    ------------------------------------------------------------------------------
    save_exposure_scan     Run an exposure scan and save the result as a NeXus
                           compliant file. This requires that libnexus and NexPy
                           are installed.
    ------------------------------------------------------------------------------

In case you are interested in the implementation of a function, you can use
:func:`.code_of`. For example::

    In [5]: code_of(pdoc)
    def pdoc(hide_blacklisted=True):
        """Render process documentation."""
        black_listed = ('show', 'start', 'init', 'rm', 'log', 'edit', 'fetch')
        field_names = ["Name", "Description"]
        table = get_default_table(field_names)
        ...

.. _pint: https://pint.readthedocs.org/en/latest/


Creating a session
==================

First of all, :ref:`initialize <init-command>` a new session::

    $ concert init new-session

and :ref:`start <edit-command>` the default editor with ::

    $ concert edit new-session

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


Importing other sessions
------------------------

To specify experiments that share a common set of devices, you can define a base
session and import it from each sub-session::

    from base import *

Now everything that was defined will be present when you start up the new
session.
