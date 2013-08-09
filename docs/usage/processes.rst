===============
Process control
===============

.. automodule:: concert.processes.base


Scanning
========

.. automodule:: concert.processes.scan
    :members:

Feedbacks
---------

Feedbacks are used to supply scan processes with a value. They must be
callables like functions or classes that implement :meth:`__call__`.

Camera
~~~~~~

.. automodule:: concert.feedbacks.camera
    :members:


Optimization
============

Optimization of a function y = f(x) can be achieved by this package.
There are different optimizers available in combination with
different algorithms. An optimizer must be used with one of the
available algorithms. E.g. a :class:`.Maximizer` used together with
a gradient feedback and  any of the optimization algorithms can be
used for focusing.

Optimizers
----------

.. automodule:: concert.optimization.optimizers
    :members:

Algorithms
----------

Optimizers can emaploy different executive algorithms.

.. automodule:: concert.optimization.algorithms
    :members:


Data processing with Ufo
========================

.. automodule:: concert.processes.ufo
    :members:


Rotation Axis Alignment
=======================

.. automodule:: concert.processes.tomoalignment
    :members:
