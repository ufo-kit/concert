Concert
=======

.. image:: https://img.shields.io/badge/Python-3.7+-blue
    :target: https://www.python.org/downloads

.. image:: https://badge.fury.io/py/concert.png
    :target: http://badge.fury.io/py/concert

.. image:: https://github.com/ufo-kit/concert/actions/workflows/tests.yml/badge.svg?branch=master
    :target: https://github.com/ufo-kit/concert/actions

.. image:: https://readthedocs.org/projects/concert/badge/?version=latest
    :target: http://concert.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/ufo-kit/concert/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/ufo-kit/concert


*Concert* is a light-weight control system interface to control Tango and native
devices. It requires Python >= 3.7, for tests you will need >= 3.8. It can be
used as a library::

    from concert.quantities import q
    from concert.devices.motors.dummy import LinearMotor

    motor = await LinearMotor()
    await motor.set_position(10 * q.mm)
    await motor.move(-5 * q.mm)

or from a session and within an integrated `IPython`_ shell::

    $ concert start

    concert > from concert.devices.motors.dummy import LinearMotor
    concert > motor = await LinearMotor()
    concert > await motor.set_position(10 * q.mm)   # The asyncio blocking way
    concert > task = motor.set_position(10 * q.mm)  # The asyncio non-blocking way
    concert > motor.position = 10 * q.mm            # User convenience for blocking calls
    concert > motor.position
    10 <Unit('millimeter')>

.. _Ipython: http://ipython.org

You can read more about *Concert* in the official `documentation`_.

.. _documentation: https://concert.readthedocs.io/en/latest/


Citation
--------

If you want to use Concert, we kindly ask you to acknowledge the respective
authorship not only by respecting the LGPL software license but also by linking
to our project website at http://ufo.kit.edu and citing the following article:
Vogelgesang, M., Farago, T., Morgeneyer, T. F., Helfen, L., dos Santos Rolo, T.,
Myagotin, A. & Baumbach, T. (2016). J. Synchrotron Rad. 23, 1254-1263,
https://doi.org/10.1107/S1600577516010195.
