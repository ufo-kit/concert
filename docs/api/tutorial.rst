===============
Getting started
===============

Get the code
============

Concert is developed using `Git`_. To clone the repository call::

    $ git clone http://ufo.kit.edu/git/concert

To get started you are encouraged to install the *development* dependencies via
pip::

    $ cd concert
    $ pip install -r requirements.txt

.. _Git: http://git-scm.com


Run the tests
-------------

The core of Concert is tested using Python's standard library :mod:`unittest`
module, the `TestFixtures`_ package and `nose`_. To run all tests, you can call
nose directly in the root directory or run make with the ``check`` argument ::

    $ make check

Some tests take a lot of time to complete and are marked with the ``@slow``
decorator. To skip them during regular development cycles, you can run ::

    $ make check-fast

You are highly encouraged to add new tests when you are adding a new feature to
the core or fixing a known bug.

.. _TestFixtures: http://pythonhosted.org/testfixtures/
.. _nose: https://nose.readthedocs.org/en/latest/


Basic concepts
==============

The core abstraction of Concert is a :class:`.Parameter`. A parameter has at
least a name but most likely also associated setter and getter callables.
Moreover, a parameter can have units and limiters associated with it.

The modules related to device creation are found here ::

    concert/
    |-- base.py
    `-- devices
        |-- base.py
        |-- cameras
        |   |-- base.py
        |   `-- ...
        |-- __init__.py
        |-- motors
        |   |-- base.py
        |   `-- ...
        `-- storagerings
            |-- base.py
            `-- ...


Adding a new device
===================

To add a new device to an existing device class (such as motor, pump,
monochromator etc.), a new module has to be added to the corresponding device
class package. Inside the new module, the concrete device class must then import
the base class, inherit from it and implement all abstract method stubs.

Let's assume we want to add a new motor called ``FancyMotor``. We first create a
new module called ``fancy.py`` in the ``concert/devices/motors`` directory
package. In the ``fancy.py`` module, we first import the base class ::

    from concert.devices.motors.base import Motor, LinearCalibration

Because a user can only set the motor position in units of meter, the device
itself must convert between motor units and meters. For this purpose, the base
class expects a :class:`.Calibration` object, such as the pre-defined
:class:`.LinearCalibration`. Now, let's sub-class :class:`.Motor`::

    class FancyMotor(Motor):
        """This is a docstring that can be looked up at run-time by the `ddoc`
        tool."""

In order to install all required parameters, we have to call the base
constructor which receives the calibration that we assume to be fixed here::

        def __init__(self):
            # 20 steps correspond to one millimeter
            calibration = LinearCalibration(20 / q.mm, 0 * q.mm)
            super(FancyMotor, self).__init__(calibration)
            self.steps = 0

Now, all that's left to do, is implementing the abstract methods that would
raise a :exc:`NotImplementedError`::

        def _get_position(self):
            return self.steps

        def _set_position(self, steps):
            self.steps = steps

.. note::

    In this motor case, the conversion from user units to steps is done before
    calling :meth:`.get_position` and :meth:`.set_position`.


Creating a device class
=======================

Defining a new device class involves adding a new package to the
``concert/devices`` directory and adding a new ``base.py`` class that inherits
from :class:`.Device` and defines all necessary :class:`.Parameter` objects and
accessor stubs.

In this exercise, we will add a new pump device class. From an abstract point of
view, a pump is characterized and manipulated in terms of the volumetric flow
rate, e.g. how many cubic millimeters per second of a medium is desired.

First, we create a new ``base.py`` into the new ``concert/devices/pumps``
directory and import everything that we need::

    import quantities as q
    from concert.base import Device, Parameter

The :class:`.Device` handles the nitty-gritty details of messaging and parameter
handling, so our base pump device must inherit from it. Furthermore, we have to
specify which kind of parameters we want to expose and how we get the
values for the parameters (by tying them to getter and setter callables)::

    class Pump(Device):
        def __init__(self):
            params = [Parameter('flow-rate',
                                fget=self._get_flow_rate,
                                fset=self._set_flow_rate,
                                unit=q.m**3 / q.s,
                                doc="Flow rate")]

            super(Pump, self).__init__(params)

        def _get_flow_rate(self):
            # This must be implemented by the actual device
            raise NotImplementedError

        def _set_flow_rate(self, value):
            # This must be implemented by the actual device
            raise NotImplementedError

.. note::

    Parameter names can only start with a letter whereas the rest of the string
    can only contain letters, numbers, dashes and underscores.

