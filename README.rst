About
=====

.. image:: https://travis-ci.org/ufo-kit/concert.png?branch=master
    :target: https://travis-ci.org/ufo-kit/concert


Concert is a light-weight control system interface to control Tango and native
devices. It can be used as a library::

    import quantities as q
    from concert.devices.motors.crio import LinearMotor

    motor = LinearMotor()
    motor.position = 10 * q.mm
    motor.move(-5 * q.mm)

or from a session and within an integrated `IPython`_ shell::

    $ concert init session
    $ concert start session
    Welcome to Concert
    This is a new session.

    In [1]: motor.position = 10 * q.mm

    In [2]: dstate
    Out[2]:
    ------------------------------------
      Name         Parameters
    ------------------------------------
      Motor         position  10.0 mm
                    state     standby
    ------------------------------------

.. _Ipython: http://ipython.org


Installation
------------

Install the latest stable version from PyPI with ``pip``::

    sudo pip install concert
