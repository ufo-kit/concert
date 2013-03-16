====
Axes
====

.. automodule:: concert.devices.axes.base

The following axis devices are available:

*   .. automodule:: concert.devices.axes.ankatango
        :members:

*   .. automodule:: concert.devices.axes.crio
        :members:


Calibration
===========

An axis has a certain notion of position depending on its type but most probably
an axis motor must be set in terms of motor steps. However, a user usually wants
to set the position in human-readable units such as meter or milli meter.

.. autoclass:: concert.devices.axes.base.Calibration
    :members:


Linear calibration
------------------

.. autoclass:: concert.devices.axes.base.LinearCalibration
