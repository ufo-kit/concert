=============
API reference
=============

Base objects
============

Parameters
----------

.. autoclass:: concert.base.Parameter
    :members:


Collection of parameters
------------------------

.. autoclass:: concert.base.Parameterizable
    :members:


Devices
-------

.. autoclass:: concert.devices.base.Device
    :show-inheritance:
    :members:


Exceptions
----------

.. autoclass:: concert.base.UnitError
.. autoclass:: concert.base.LimitError
.. autoclass:: concert.base.ParameterError
.. autoclass:: concert.base.ReadAccessError
.. autoclass:: concert.base.WriteAccessError


Calibration
-----------

.. autoclass:: concert.devices.base.Calibration
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.base.LinearCalibration
    :show-inheritance:
    :members:


Base devices
============


Cameras
-------

.. autoclass:: concert.devices.cameras.base.Camera
    :show-inheritance:
    :members:


I/O
---

.. autoclass:: concert.devices.io.base.Port
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.io.base.IO
    :show-inheritance:
    :members:


Monochromators
--------------

.. autoclass:: concert.devices.monochromators.base.Monochromator
    :show-inheritance:
    :members:


Motors
------

.. autoclass:: concert.devices.motors.base.Motor
    :show-inheritance:
    :members:


Pumps
-----

.. autoclass:: concert.devices.pumps.base.Pump
    :show-inheritance:
    :members:


Scales
------

.. autoclass:: concert.devices.scales.base.Scales
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.scales.base.TarableScales
    :show-inheritance:
    :members:


Shutters
--------

.. autoclass:: concert.devices.shutters.base.Shutter
    :show-inheritance:
    :members:


Storage rings
-------------

.. autoclass:: concert.devices.storagerings.base.StorageRing
    :show-inheritance:
    :members:


Helpers
=======

.. automodule:: concert.helpers
    :members:

.. automodule:: concert.coroutines
    :members:


Coroutines
==========

Coroutines provide a way to process data and yield execution until more data
is produced. To build flexible coroutine-based processing pipelines in Python,
the enhanced ``yield`` statement is used. To simplify startup of the coroutine,
you can decorate a function with :py:func:`.coroutine`::

    from concert.helpers import coroutine

    @coroutine
    def printer():
        while True:
            item = yield
            print(item)

Sinks
-----

Sinks are the simpler version of coroutines where one processes the data but
doesn't forward them, thus a sink is a pipeline terminal.

.. automodule:: concert.coroutines.sinks
    :members:

Filters
-------

Because the ``printer`` only consumes data it is an end point, hence called a
sink. Filters on the other hand hook into the stream and turn the input into
some output. For example, to generate a stream of squared input, you would
write::

    @coroutine
    def square(target):
        while True:
            item = yield
            target.send(item**2)

.. automodule:: concert.coroutines.filters
    :members:


Sessions
========

.. automodule:: concert.session.utils
    :members:


Processes
=========

.. automodule:: concert.processes
    :members:


Optimization
============

.. automodule:: concert.optimization
    :members:


Networking
==========

.. automodule:: concert.networking


Socket Connections
------------------

.. autoclass:: SocketConnection
    :members:

.. autoclass:: Aerotech
    :members:


TANGO
-----

.. _Tango: http://www.tango-controls.org/
.. _PyTango: http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
.. _DeviceProxy: http://www.tango-controls.org/static/PyTango/latest/doc/html/client/device_proxy.html

Tango_ devices are interfaced by PyTango_, one can obtain the DeviceProxy_ by
the :py:func:`get_tango_device` function.

.. autofunction:: get_tango_device


Extensions
==========

.. automodule:: concert.ext.ufo
    :members:
