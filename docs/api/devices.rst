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


Motors
======

.. autoclass:: concert.devices.motors.base.Motor
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.motors.base.Calibration
    :show-inheritance:
    :members:


Cameras
=======

.. autoclass:: concert.devices.cameras.base.Camera
    :show-inheritance:
    :members:


Monochromators
==============

.. autoclass:: concert.devices.monochromators.base.Monochromator
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
