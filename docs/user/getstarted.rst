.. _tutorial:

========
Tutorial
========

Concert is primarily a user interface to control devices commonly found at a
Synchrotron beamline. This guide will briefly show you how to use and extend it.


Running a session
=================

In case you don't have a beamline at hand, you can import our sample sessions
with the :ref:`import <import-command>` command::

    $ concert import --repo https://github.com/ufo-kit/concert-examples

Now :ref:`start <start-command>` the tutorial session::

    $ concert start tutorial

You will be greeted by an IPython shell loaded with pre-defined devices,
processes and utilities like the pint_ package for unit calculation. Although,
this package is primarily used for talking to devices, you can also use it to do
simple calculations::

    tutorial > a = 9.81 * q.m / q.s**2
    tutorial > "Velocity after 5 seconds: {0}".format(5 * q.s * a)

    'Velocity after 5 seconds: 49.05 meter / second'

You can get an overview of all defined devices by calling the 
:func:`~.ddoc` function::

    tutorial > ddoc()

    ------------------------------------------------------------------------------
      Name         Description   Parameters
    ------------------------------------------------------------------------------
      motor        None           Name      Access  Unit  Description
                                  position  rw      m     Position of the motor
    ------------------------------------------------------------------------------
      ...

Now, by typing just the name of a device, you can see it's currently set parameter
values::

    tutorial > motor

    <concert.devices.motors.dummy.LinearMotor object at 0x9419f0c>
     Parameter  Value                    
     position   12.729455653 millimeter

To get an overview of all devices' parameter values, use the :func:`~.dstate`
function::

    tutorial > dstate()

    ---------------------------------------------
      Name         Parameters
    ---------------------------------------------
      motor        position  99.382 millimeter
    ---------------------------------------------
      ...

To change the value of a parameter, you simply assign a new value to it::

    tutorial > motor.position = 2 * q.mm

Now, check the position to verify that the motor reached the target position::

    tutorial > motor.position
    <Quantity(2.0, 'millimeter')>

Depending on the device, changing a parameter will block as long as the device
has not yet reached the final target state. You can read more about asynchronous
execution in the :ref:`controlling-devices` chapter.

.. note::

    A parameter value is always checked for the correct unit and soft limit
    condition. If you get an error, check twice that you are using a compatible
    unit (setting two seconds on a motor position is obviously not) and are
    within the allowed parameter range.

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

    tutorial > code_of(code_of)
    def code_of(func):
        """Show implementation of *func*."""
        source = inspect.getsource(func)

        try:
            ...

.. note::

    Because we are actually running an IPython shell, you can _always_
    tab-complete objects and attributes. For example, to change the motor
    position to 1 millimeter, you could simply type ``mot<Tab>.p<Tab> = 1 * q.mm``.

.. _pint: https://pint.readthedocs.org/en/latest/


Creating a session
==================

First of all, :ref:`initialize <init-command>` a new session::

    $ concert init new-session

and :ref:`start <edit-command>` the default editor with ::

    $ concert edit new-session

At the top of the file, you can see a string enclosed in three ``"``. This
should be changed to something descriptive as it will be shown each time you start
the session.


Adding devices
--------------

To create a device suited for your experiment you have to import it first.
Concert uses the following packaging scheme to separate device classes and
device implementations: ``concert.devices.[class].[implementation]``. Thus if
you want to create a dummy ring from the storage ring class, you would add this
line to your session::

    from concert.devices.storagerings.dummy import StorageRing

Once imported, you can create the device and give it a name that will be
accessible from the command line shell::

    from concert.devices.motors.dummy import LinearMotor

    ring = StorageRing()
    motor = LinearMotor()


Importing other sessions
------------------------

To specify experiments that share a common set of devices, you can define a base
session and import it from each sub-session::

    from base import *

Now everything that was defined will be present when you start up the new
session.
