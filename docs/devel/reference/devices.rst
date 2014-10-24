Device classes
==============

Cameras
-------

.. automodule:: concert.devices.cameras.base
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.cameras.uca.Camera
.. autoclass:: concert.devices.cameras.pco.Pco
.. autoclass:: concert.devices.cameras.pco.Dimax
.. autoclass:: concert.devices.cameras.pco.PCO4000

.. autoclass:: concert.devices.cameras.dummy.Camera


Grippers
--------

.. automodule:: concert.devices.grippers.base
    :show-inheritance:
    :members:


I/O
---

.. autoclass:: concert.devices.io.base.Signal
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.io.base.IO
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.io.dummy.IO


Lightsources
--------------

.. autoclass:: concert.devices.lightsources.base.LightSource
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.lightsources.dummy.LightSource


Monochromators
--------------

.. autoclass:: concert.devices.monochromators.base.Monochromator
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.monochromators.dummy.Monochromator


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

.. autoclass:: concert.devices.motors.dummy.LinearMotor
.. autoclass:: concert.devices.motors.dummy.ContinuousLinearMotor

Rotational
~~~~~~~~~~

Rotational motors are characterized by rotating around an axis.

.. autoclass:: concert.devices.motors.base.RotationMotor
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.motors.base.ContinuousRotationMotor
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.motors.dummy.RotationMotor
.. autoclass:: concert.devices.motors.dummy.ContinuousRotationMotor


Axes
----

An axis is a coordinate system axis which can realize either translation
or rotation, depending by which type of motor it is realized.

.. autoclass:: concert.devices.positioners.base.Axis
    :show-inheritance:
    :members:


Photodiodes
-----------

Photodiodes measure light intensity.

.. autoclass:: concert.devices.photodiodes.base.PhotoDiode
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.photodiodes.dummy.PhotoDiode


Positioners
-----------

Positioner is a device consisting of more
:py:class:`concert.devices.positioners.base.Axis`
instances which make it possible to specify a 3D position and
orientation of some object.

.. autoclass:: concert.devices.positioners.base.Positioner
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.positioners.dummy.Positioner


Imaging Positioners
~~~~~~~~~~~~~~~~~~~

Imaging positioner is a positioner capable of moving in *x* and *y*
directions by the given amount of pixels.

.. autoclass:: concert.devices.positioners.imaging.Positioner
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.positioners.dummy.ImagingPositioner


Pumps
-----

.. autoclass:: concert.devices.pumps.base.Pump
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.pumps.dummy.Pump


Sample changers
---------------

.. autoclass:: concert.devices.samplechangers.base.SampleChanger
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

.. autoclass:: concert.devices.scales.dummy.Scales


Shutters
--------

.. autoclass:: concert.devices.shutters.base.Shutter
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.shutters.dummy.Shutter


Storage rings
-------------

.. autoclass:: concert.devices.storagerings.base.StorageRing
    :show-inheritance:
    :members:

.. autoclass:: concert.devices.storagerings.dummy.StorageRing
