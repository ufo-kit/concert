Extensions
==========

Concert integrates third-party software in the ``ext`` package. Because the
dependencies of these modules are not listed as Concert dependencies, you have
to make sure, that the appropriate libraries and modules are installed.


UFO Processing
--------------

.. automodule:: concert.ext.ufo
    :members:


.. _viewers:

Viewers
-------

.. automodule:: concert.ext.viewers
    :members:


Sample detection
----------------

Remote sample detection is implemented by
:class:`concert.experiments.addons.tango.SampleDetector` and the
``TangoSampleDetect`` device server. The server accepts single encoded images
or a ZMQ image stream, converts input to 8-bit three-channel data, and runs an
Ultralytics YOLO model. Bounding-box changes may be published as JSON messages
with the ``sample-bbox`` key.

The server-side dependencies ``torch`` and ``ultralytics`` are optional and
must be installed separately.
