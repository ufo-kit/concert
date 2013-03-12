====
Axes
====

.. module:: concert.devices.axes.base

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


Linear calibration
------------------

.. autoclass:: LinearCalibration


Implementations
===============

.. automodule:: concert.devices.axes.crio
    :members:
