===============
Process control
===============

The main aspect of an experimental control system is to manage devices in a
meaningful way.

Concert devices can be changed asynchronously. The same is true for processes,
so that independent processes can run at the same time. However, processes that
access the same devices concurrently could potentially clash. Therefore, two
processes are serialized from an outside point of view but run
asynchronously each on its own.


Scanning
========

.. automodule:: concert.processes.scan
    :members:
