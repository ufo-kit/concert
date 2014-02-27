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


Helpers
=======

.. automodule:: concert.helpers
    :members:
