====
Axes
====

.. module:: control.devices.axes.axis

Every control object that moves along single one-dimensional direction is an
:class:`Axis`:

.. autoclass:: Axis
    :members:


Calibration
===========

An axis has a certain notion of position depending on its type but most probably
an axis motor must be set in terms of motor steps. However, a user usually wants
to set the position in human-readable units such as meter or milli meter.

.. autoclass:: Calibration
    :members:


Defined calibrations
--------------------

.. autoclass:: LinearCalibration


Implementations
===============

.. automodule:: control.devices.axes.crio
    :members:

