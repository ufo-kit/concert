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


Base devices
============


Cameras
-------

.. autoclass:: concert.devices.cameras.base.Camera
    :show-inheritance:
    :members:


I/O
---

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

.. autoclass:: concert.devices.motors.base.PositionMixin
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.motors.base.ContinuousMixin
    :show-inheritance:
    :members:

Linear
~~~~~~

Linear motors are characterized by moving along a straight line.

.. autoclass:: concert.devices.motors.base.LinearMotor
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.motors.base.ContinuousLinearMotor
    :show-inheritance:
    :members:


Rotational
~~~~~~~~~~

Rotational motors are characterized by rotating around an axis.

.. autoclass:: concert.devices.motors.base.RotationMotor
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.motors.base.ContinuousRotationMotor
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


Sinks
-----

.. automodule:: concert.coroutines.sinks
    :members:


Filters
-------

.. automodule:: concert.coroutines.filters
    :members:


Asynchronous execution
----------------------

.. automodule:: concert.async
    :members:


Configuration
-------------

.. automodule:: concert.config
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
