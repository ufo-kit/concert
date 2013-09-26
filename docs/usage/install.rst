============
Installation
============

It is recommended to use pip_ for installing Concert. The fastest way to install
it is from PyPI::

    $ sudo pip install concert

This will install the latest stable version. If you prefer an earlier stable
version, you can fetch a tarball and install with::

    $ sudo pip install concert-x.y.z.tar.gz

If you haven't have pip_ available, you can extract the tarball and install using
the supplied ``setup.py`` script::

    $ tar xfz concert-x.y.z.tar.gz
    $ cd concert-x.y.z
    $ sudo python setup.py install

More information on installing Concert using the ``setup.py`` script, can be
found in the official `Python documentation`__.

To install the Concert from the current source, follow the instructions given in
the :ref:`developer documentation <get-the-code>`.

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


Getting started
===============

You can now run the ``concert`` binary to enter the IPython command-line shell
for manipulating devices. There won't be much to see, but if you just want to
play a bit you can fetch some example sessions with ::

    $ concert fetch --repo https://github.com/ufo-kit/concert-examples

and start the provided sessions with ::

    $ concert start zig-zag-scan
