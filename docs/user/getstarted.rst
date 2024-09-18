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

How to execute more things concurrently and how to stop execution can be found
in :ref:`concurrent-execution`.


Creating a session
==================

First of all, :ref:`initialize <init-command>` a new session::

    $ concert init new_session

and :ref:`start <edit-command>` the default editor with ::

    $ concert edit new_session

At the top of the file, you can see a string enclosed in three ``"``. This
should be changed to something descriptive as it will be shown each time you start
the session.

Sessions are just normal Python modules with one additional feature that you may
use top-level ``await`` in them, i.e. outside of an ``async def`` function, for
more info see :ref:`shell-importing`.

In the session code you can specify whether a session can run only once or
several times, use ``MULTIINSTANCE = True | False`` for specifying this. If the
``MULTIINSTANCE`` assignment is not found it is ``False`` by default, which
means the session can run only once (have one instance). This is useful for
experiment sessions which use various resources and set up remote connections.


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

    ring = await StorageRing()
    motor = await LinearMotor()


Importing other sessions
------------------------

To specify experiments that share a common set of devices, you can define a base
session and import it from each sub-session::

    from base import *

Now everything that was defined will be present when you start up the new
session.


Hello World
===========

Let's create a session::

    concert edit scan

And then add some code inside so that we can discuss some of the core Concert
features. You can download the scan_ example or just copy this::

    """# *scan* shows scanning of camera's exposure time.

    ## Usage
        await run(producer, line, acc)

    ## Notes
    """

    import asyncio
    import logging
    from inspect import iscoroutinefunction
    import concert
    concert.require("0.30.0")

    from concert.coroutines.base import broadcast
    from concert.coroutines.sinks import Accumulate
    from concert.quantities import q
    from concert.session.utils import cdoc, ddoc, dstate, pdoc, code_of
    from concert.devices.cameras.dummy import Camera
    from concert.ext.viewers import PyplotViewer, PyQtGraphViewer
    from concert.processes.common import ascan

    LOG = logging.getLogger(__name__)
    # Disable progress bar in order not to interfere with printing
    concert.config.PROGRESS_BAR = False


    async def feedback():
        """Our feedback just returns image mean."""
        # Let's pretend this is a serious operation which takes a while
        await asyncio.sleep(1)
        image = await camera.grab()
        # Also show the current image
        await viewer.show(image)

        return image.mean()


    async def run(producer, line, accumulator):
        coros = broadcast(producer, line, accumulator)
        await asyncio.gather(*coros)

        return accumulator.items


    viewer = await PyQtGraphViewer()
    # The last image will be quite bright
    viewer.limits = 0, 10000
    # Plot image mean
    line = await PyplotViewer(style='-o')
    # Dummy camera
    camera = await Camera()
    # For scan results collection
    acc = Accumulate()
    # Let's create a scan so that it can be directly plugged into *run*
    producer = ascan(camera['exposure_time'], 1 * q.ms, 100 * q.ms, 10 * q.ms, feedback=feedback)

With this code you can execute the scan showing both the image and the mean and
storing the result in :data:`acc` by::

    items = await run(producer, line, acc)
    print(items) # or print(acc.items)
    # Gives
    [(1 <Unit('millisecond')>, 101.01860026041666),
     (11 <Unit('millisecond')>, 1101.0648697916668),
     (21 <Unit('millisecond')>, 2101.0111751302084),
     (31 <Unit('millisecond')>, 3100.9252408854168),
     (41 <Unit('millisecond')>, 4101.011533203125),
     (51 <Unit('millisecond')>, 5101.0090625),
     (61 <Unit('millisecond')>, 6100.966005859375),
     (71 <Unit('millisecond')>, 7101.112858072916),
     (81 <Unit('millisecond')>, 8100.928743489583),
     (91 <Unit('millisecond')>, 9101.179690755209)]

or you can simply run the scan showing both the image and the mean to see the
mean::

    await line(producer)

or you can iterate through the values and decide what to do with them yourself::

    async for x, y in producer:
        print(f'x={x}, y={y}')
    # Gives
    x=1 millisecond, y=101.00574544270833
    x=11 millisecond, y=1100.9828515625
    x=21 millisecond, y=2100.9941015625
    x=31 millisecond, y=3100.982431640625
    x=41 millisecond, y=4100.772060546875
    x=51 millisecond, y=5100.855152994792
    x=61 millisecond, y=6100.988649088542
    x=71 millisecond, y=7101.148798828125
    x=81 millisecond, y=8101.085227864583
    x=91 millisecond, y=9100.949088541667


.. _scan: https://github.com/ufo-kit/concert-examples/blob/master/scan.py
