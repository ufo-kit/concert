===================
Controlling devices
===================

.. automodule:: concert.base


Device classes
==============

.. toctree::
    :maxdepth: 2

    axis
    camera


Interface
=========

.. autoclass:: concert.base.ConcertObject
    :members:


Implementing objects
--------------------

.. py:class:: ConcertObject

    .. py:method:: _register(param, getter, setter, unit, limiter=None)

        Registers a parameter name `param`.

    .. note::

        :meth:`_register` can be called several times along the inheritance
        hierarchy. Each time a new setter is registered with the same name, the
        setter will be applied in *reverse* order. That means if ``A`` inherits from
        ``Device`` and ``B`` inherits from ``A``, calling ``set`` on an object of
        type ``B`` will actually call ``B.set(A.set(x))``.
