===============
Process control
===============

.. automodule:: concert.processes.base


Scanning
========

.. automodule:: concert.processes.scan
    :members:


Optimization
============

Optimization is a procedure to iteratively find the best possible match to y = f(x).

Execution
----------------------

.. automodule:: concert.optimization.base
    :members:

Algorithms
----------

.. automodule:: concert.optimization.algorithms
    :members:

Scalar Optimizers
-----------------
.. automodule:: concert.optimization.scalar
    :members:


Feedbacks
---------

Feedbacks are used to supply scan processes with a value. They must be
callables like functions or classes that implement :meth:`__call__`.

Camera
~~~~~~

.. automodule:: concert.feedbacks.camera
    :members:

Data processing with Ufo
========================

.. automodule:: concert.processes.ufo
    :members:


Rotation Axis Alignment
=======================

.. automodule:: concert.processes.tomoalignment
    :members:
