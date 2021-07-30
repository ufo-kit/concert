===============
Writing devices
===============

.. _get-the-code:

Get the code
============

Concert is developed using `Git`_ on the popular GitHub platform. To clone the
repository call::

    $ git clone https://github.com/ufo-kit/concert

To get started you are encouraged to install the *development* dependencies via
pip::

    $ cd concert
    $ sudo pip install -r requirements.txt

After that you can simply install the development source with ::

    $ sudo make install

.. _Git: http://git-scm.com


Run the tests
-------------

The core of Concert is tested using Python's standard library :mod:`unittest`
module and `nose`_. To run all tests, you can call nose directly in the root
directory or run make with the ``check`` argument ::

    $ make check

Some tests take a lot of time to complete and are marked with the ``@slow``
decorator. To skip them during regular development cycles, you can run ::

    $ make check-fast

You are highly encouraged to add new tests when you are adding a new feature to
the core or fixing a known bug.

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

    from concert.devices.motors.base import LinearMotor

Our motor will be a linear one, let's sub-class :class:`~.base.LinearMotor`::

    class FancyMotor(LinearMotor):
        """This is a docstring that can be looked up at run-time by the `ddoc`
        tool."""

In order to install all required parameters, we have to call the base
constructor. Now, all that's left to do, is implementing the abstract methods that
would raise a :exc:`.AccessorNotImplementedError`::

        def _get_position(self):
            # the returned value must have units compatible with units set in
            # the Quantity this getter implements
            return self.position

        def _set_position(self, position):
            # position is guaranteed to be in the units set by the respective
            # Quantity
            self.position = position

We guarantee that in setters which implement a :class:`.Quantity`, like the
:meth:`._set_position` above, obtain the value in the exact same units as they
were specified in the respective :class:`.Quantity` they implement. E.g. if the
above :meth:`_set_position` implemented a quantity with units set in kilometers,
the :attr:`~.LinearMotor.position` of the :meth:`._set_position` will also be in
kilometers.  On the other hand the getters do not need to return the exact same
quantity but the value must be compatible, so the above :meth:`._get_position`
could return millimeters and the user would get the value in kilometers, as
defined in the respective :class:`.Quantity`.

Parameter setters can be cancelled by hitting *ctrl-c*. If you want a parameter to
make some cleanup action after *ctrl-c* is pressed, you should implement the
``_cancel_param`` method in the device class, for the motor above you can write::

        def _cancel_position(self):
            # send stop command

And you are guaranteed that when you interrupt the setter the motor stops
moving.


Creating a device class
=======================

Defining a new device class involves adding a new package to the
``concert/devices`` directory and adding a new ``base.py`` class that inherits
from :class:`.Device` and defines necessary :class:`.Parameter` and
:class:`.Quantity` objects.

In this exercise, we will add a new pump device class. From an abstract point of
view, a pump is characterized and manipulated in terms of the volumetric flow
rate, e.g. how many cubic millimeters per second of a medium is desired.

First, we create a new ``base.py`` into the new ``concert/devices/pumps``
directory and import everything that we need::

    from concert.quantities import q
    from concert.base import Quantity
    from concert.devices.base import Device

The :class:`.Device` handles the nitty-gritty details of messaging and parameter
handling, so our base pump device must inherit from it. Furthermore, we have to
specify which kind of parameters we want to expose and how we get the
values for the parameters (by tying them to getter and setter callables)::

    class Pump(Device):

        flow_rate = Quantity(q.m**3 / q.s,
                             lower=0 * q.m**3 / q.s, upper=1 * q.m**3 / q.s,
                             help="Flow rate of the pump")

        def __init__(self):
            super(Pump, self).__init__()

The `flow_rate` parameter can only receive values from zero to one cubic meter
per second.

We didn't specify explicit *fget* and *fset* functions, which is why  implicit
setters and getters called `_set_flow_rate` and `_get_flow_rate` are installed.
The real devices then need to implement these. You can however, also specify
explicit setters and getters in order to hook into the get and set process::

    class Pump(Device):

        def __init__(self):
            super(Pump, self).__init__()

        def _intercept_get_flow_rate(self):
            return self._get_flow_rate() * 10

        flow_rate = Quantity(q.m**3 / q.s,
                             fget=_intercept_get_flow_rate)

Be aware, that in this case you have to list the parameter *after* the functions
that you want to refer to.

In case you want to specify the name of the accessor function yourself and rely
on implementation by subclasses, you have to raise an
:exc:`.AccessorNotImplementedError`::

    from concert.base import AccessorNotImplementedError

    class Pump(Device):

        ...

        def _set_flow_rate(self, flow_rate):
            raise AccessorNotImplementedError


State machine
-------------

A formally defined finite state machine is necessary to ensure and reason about
correct behaviour. Concert provides an implicitly defined, decorator-based state
machine. The machine can be used to model devices which support hardware state
reading but also the ones which don't, thanks to the possibility to store the
state in the device itself. To use the state machine you need to declare a
:class:`.State` object in the base device class and apply the :func:`.check`
decorator on each method that changes the state of a device.  If you are
implementing a device which can read the hardware state you need to define the
``_get_state`` method. If you are implementing a device which does not support
hardware state reading then you need to redefine the :class:`.State` in such a
way that it has a default value (see the code below) and you can ensure it is
changed by respective methods by using the :func:`.transition` decorator on such
methods, so that you can keep track of state changes at least in software and
comply with transitioning. Examples of such devices could look as follows::

    from concert.base import Quantity, State, transition, check


    class BaseMotor(Device):

        """A base motor class."""

        state = State()
        position = Quantity(q.m)

        @check(source='standby', target='moving')
        def start(self):
            ...

        def _start(self):
            # the actual implementation of starting something
            ...


    class Motor(BaseMotor):

        """A motor with hardware state reading support."""

        ...

        def _start(self):
            # Implementation communicates with hardware
            ...

        def _get_state(self):
            # Get the state from the hardware
            ...


    class StatelessMotor(BaseMotor):

        """A motor which doesn't support state reading from hardware."""

        # we have to specify a default value since we cannot get it from
        # hardware
        state = State(default='standby')

        ...

        @transition(target='moving')
        def _start(self):
            ...

The example above explains two devices with the same functionality, however, one
supports hardware state reading and the other does not. When they want to
``start`` the state is checked before the method is executed and afterwards. By
checking we mean the current state is checked against the one specified by
``source`` and the state after the execution is checked against ``target``.  The
``Motor`` represents a device which supports hardware state reading.  That means
all we have to do is to implement ``_get_state``. The ``StatelessMotor``, on the
other hand, has no way of determining the hardware state, thus we need to keep
track of it in software. That is achieved by the :func:`.transition` which sets the
device state after the execution of the decorated function to ``target``.  This
way the ``start`` method can look the same for both devices.

Besides single state strings you can also add lists of strings and a catch-all
``*`` state that matches all states.

There is no explicit error handling implemented for devices which support
hardware state reading but it can be easily modeled by adding error states and
reset functions that transition out of them. In case the device does not support
state reading and it runs into an error state all you need to do is to raise a
:class:`.StateError` exception, which has a parameter ``error_state``. The
exception is caught by :func:`.transition` and the ``error_state`` parameter is used
for setting the device state.


Parameters
~~~~~~~~~~

In case changing a parameter value causes a state transition, add a
:func:`.check` to the :class:`.Quantity` object or to the :class:`.Parameter` object::

    class Motor(Device):

        state = State(default='standby')

        velocity = Quantity(q.m / q.s,
                            check=check(source='*', target='moving'))

        foo = Parameter(check=check(source='*', target='*'))


Commands
========

Concert's *ctrl-k* cancels coroutines defined in Concert, the current session
and in the :data:`.ABORTABLE_PATHS` list. If you want your library to be
cancellable by *ctrl-k*, you should add this line to your ``__init__.py``::

    concert.session.utils.ABORTABLE_PATHS.append(__file__)
