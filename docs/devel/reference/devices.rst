Device classes
==============

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


Axes
----

An axis is a coordinate system axis which can realize either translation
or rotation, depending by which type of motor it is realized.

.. autoclass:: concert.devices.positioners.base.Axis
    :show-inheritance:
    :members:


Positioners
-----------

Positioner is a device consisting of more
:py:class:`concert.devices.positioners.base.Axis`
instances which make it possible to specify a 3D position and
orientation of some object.

.. autoclass:: concert.devices.positioners.base.Positioner
    :show-inheritance:
    :members:


Imaging Positioners
~~~~~~~~~~~~~~~~~~~

Imaging positioner is a positioner capable of moving in *x* and *y*
directions by the given amount of pixels.

.. autoclass:: concert.devices.positioners.imaging.Positioner
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


