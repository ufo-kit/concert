Extensions
==========

Concert integrates third-party software in the ``ext`` package. Because the
dependencies of these modules are not listed as Concert dependencies, you have
to make sure, that the appropriate libraries and modules are installed.


UFO Processing
--------------

Base objects
~~~~~~~~~~~~

.. autoclass:: concert.ext.ufo.PluginManager
    :members:

.. autoclass:: concert.ext.ufo.InjectProcess
    :members:


Coroutines
~~~~~~~~~~

.. autoclass:: concert.ext.ufo.Backproject
    :show-inheritance:
    :members:


Viewers
-------

.. automodule:: concert.ext.viewers
    :members:
