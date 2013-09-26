=================================
Application Programming Interface
=================================

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
===========

.. autoclass:: concert.devices.base.Calibration
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.base.LinearCalibration
    :show-inheritance:
    :members:


Motors
======

.. autoclass:: concert.devices.motors.base.Motor
    :show-inheritance:
    :members:


Cameras
=======

.. autoclass:: concert.devices.cameras.base.Camera
    :show-inheritance:
    :members:


I/O
===

.. autoclass:: concert.devices.io.base.Port
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.io.base.IO
    :show-inheritance:
    :members:


Monochromators
==============

.. autoclass:: concert.devices.monochromators.base.Monochromator
    :show-inheritance:
    :members:


Pumps
=====

.. autoclass:: concert.devices.pumps.base.Pump
    :show-inheritance:
    :members:


Scales
======

.. autoclass:: concert.devices.scales.base.Scales
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.scales.base.TarableScales
    :show-inheritance:
    :members:


Shutters
========

.. autoclass:: concert.devices.shutters.base.Shutter
    :show-inheritance:
    :members:


Storage rings
=============

.. autoclass:: concert.devices.storagerings.base.StorageRing
    :show-inheritance:
    :members:
