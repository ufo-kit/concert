Core objects
============

Parameters
----------

.. autoclass:: concert.base.Parameter
    :members:

.. autoclass:: concert.base.ParameterValue
    :members:

.. autoclass:: concert.base.Quantity
    :show-inheritance:
    :members:

.. autoclass:: concert.base.QuantityValue
    :show-inheritance:
    :members:


Collection of parameters
------------------------

.. autoclass:: concert.base.Parameterizable
    :members:


State machine
-------------

.. autoclass:: concert.base.State
    :members:

.. autofunction:: concert.base.check

.. autofunction:: concert.base.transition


Devices
-------

.. autoclass:: concert.devices.base.Device
    :show-inheritance:
    :members:


Asynchronous execution
----------------------

.. automodule:: concert.async
    :members:


Exceptions
----------

.. autoclass:: concert.base.UnitError
.. autoclass:: concert.base.LimitError
.. autoclass:: concert.base.ParameterError
.. autoclass:: concert.base.AccessorNotImplementedError
.. autoclass:: concert.base.ReadAccessError
.. autoclass:: concert.base.WriteAccessError
.. autoclass:: concert.base.StateError


Configuration
-------------

.. automodule:: concert.config
    :members:


Sessions
========

.. automodule:: concert.session.utils
    :members:


Networking
==========

Networking package facilitates all network connections, e.g. sockets and Tango.


Socket Connections
------------------

.. autoclass:: concert.networking.base.SocketConnection
    :members:

.. autoclass:: concert.networking.aerotech.Connection
    :members:


TANGO
-----

.. _Tango: http://www.tango-controls.org/
.. _PyTango: http://www.esrf.eu/computing/cs/tango/tango_doc/kernel_doc/pytango/latest/index.html
.. _DeviceProxy: http://www.esrf.eu/computing/cs/tango/tango_doc/kernel_doc/pytango/latest/client_api/device_proxy.html

Tango_ devices are interfaced by PyTango_, one can obtain the DeviceProxy_ by
the :py:func:`get_tango_device` function.

.. autofunction:: concert.networking.base.get_tango_device


Helpers
=======

.. automodule:: concert.helpers
    :members:
