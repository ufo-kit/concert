============
Installation
============

It is recommended to use pip for installing Concert. Once, a release tarball was
downloaded it can be installed system-wide with::

    $ sudo pip install concert-x.y.z.tar.gz

If you haven't have pip_ available, you can extract the tarball and install using
this method::

    $ tar xfz concert-x.y.z.tar.gz
    $ cd concert-x.y.z
    $ sudo python setup.py install

To get the latest source follow the instructions given in the :ref:`developer
documentation <get-the-code>`.

More information on installing Concert using the ``setup.py`` script, can be
found in the official `Python documentation`__.

__ http://docs.python.org/2/install/index.html


Installing into a virtualenv
============================

It is sometimes a good idea to install third-party Python modules independent of
the system installation. This can be achieved easily using pip_ and virtualenv_.
When virtualenv is installed, create a new empty environment and activate that
with ::

    $ virtualenv my_new_environment
    $ . my_new_environment/bin/activate

Now, you can install Concert's requirements and Concert itself ::

    $ pip install -e path_to_concert/

As long as ``my_new_environment`` is active, you can use Concert.


.. _pip: https://pypi.python.org/pypi
.. _virtualenv: http://virtualenv.org
