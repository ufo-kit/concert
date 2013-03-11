=======
Devices
=======

.. module:: control.devices.device

A devices is the fundamental base of a controllable object. It provides a
mechanism to set and get arbitrary parameters that can have an associated
real-world unit. The following base devices are available:

.. toctree::
    :maxdepth: 2

    axis


Programming interface
=====================

.. autoclass:: Device
    :members:

A device is also a control object that automatically uses the global
:py:data:`control.events.dispatcher.dispatcher` object to send and subscribe
messages to which interested parties can subscribe to:

.. autoclass:: control.controlobject.ControlObject
    :members:


Implementing devices
====================

.. py:class:: Device

    .. py:method:: _register(param, getter, setter, unit, limiter=None)

        Registers a parameter name `param`.

    .. note::

        :meth:`_register` can be called several times along the inheritance
        hierarchy. Each time a new setter is registered with the same name, the
        setter will be applied in *reverse* order. That means if ``A`` inherits from
        ``Device`` and ``B`` inherits from ``A``, calling ``set`` on an object of
        type ``B`` will actually call ``B.set(A.set(x))``.

