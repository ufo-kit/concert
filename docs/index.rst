=======
Concert
=======

About
=====

.. include:: ../README.rst


Installation
============

Concert uses distutils, so in case you downloaded a source tarball, you can
simply install it system-wide using::

    $ tar xfz concert-x.y.z.tar.gz
    $ cd concert-x.y.z
    $ sudo python setup.py install

More information on installing Concert using the ``setup.py`` script, can be
found in the official `Python documentation`__.

__ http://docs.python.org/2/install/index.html


Installing into a virtualenv
----------------------------

It is sometimes a good idea to install third-party Python modules independent of
the system installation. This can be achieved easily using pip_ and
virtualenv_. Once pip and virtualenv is installed, create a new empty
environment and activate that with ::

    $ virtualenv my_new_environment
    $ . my_new_environment/bin/activate

Now, you can install Concert's requirements and Concert itself ::

    $ pip install -r path_to_concert/requirements.txt
    $ pip install -e path_to_concert/

As long as ``my_new_environment`` is active, you can use Concert.


.. _pip: https://pypi.python.org/pypi
.. _virtualenv: http://virtualenv.org


User documentation
==================

.. toctree::
    :maxdepth: 2
    :glob:

    usage/sessions


Developer documentation
=======================
.. toctree::
    :maxdepth: 2

    usage/concertobject
    usage/dispatcher
    usage/axis
    usage/camera
