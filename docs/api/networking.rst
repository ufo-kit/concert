==========
Networking
==========

.. automodule:: concert.networking


Socket Connections
------------------

.. autoclass:: SocketConnection
    :members:

.. autoclass:: Aerotech
    :members:


TANGO
-----

.. _Tango: http://www.tango-controls.org/
.. _PyTango: http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
.. _DeviceProxy: http://www.tango-controls.org/static/PyTango/latest/doc/html/client/device_proxy.html

Tango_ devices are interfaced by PyTango_, one can obtain the DeviceProxy_ by
the :py:func:`get_tango_device` function.

.. autofunction:: get_tango_device

