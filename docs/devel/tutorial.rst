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
    $ pip install -r requirements.txt

After that you can simply install the development source with ::

    $ make install

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
constructor. Moreover, we need to set the conversion of every :class:`.Quantity`
which belong to our new device, in our case the ``position``::

        def __init__(self):
            super(FancyMotor, self).__init__()
            # 20 steps correspond to one millimeter
            self['position'].conversion = lambda x: x * 20 * q.count / q.mm

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
    from concert.base import Parameter
    from concert.devices.base import Device

The :class:`.Device` handles the nitty-gritty details of messaging and parameter
handling, so our base pump device must inherit from it. Furthermore, we have to
specify which kind of parameters we want to expose and how we get the
values for the parameters (by tying them to getter and setter callables)::

    class Pump(Device):

        flow_rate = Parameter(unit=q.m**3 / q.s)

        def __init__(self):
            super(Pump, self).__init__()

This installs implicit setters and getters called `_set_flow_rate` and
`_get_flow_rate` that need to be implemented by the real devices. You can
however, also specify explicit setters and getters in order to hook into the get
and set process::

    class Pump(Device):

        def __init__(self):
            super(Pump, self).__init__()

        def _intercept_get_flow_rate(self):
            return self._get_flow_rate() * 10

        flow_rate = Parameter(unit=q.m**3 / q.s,
                              fget=_intercept_get_flow_rate)

Be aware, that in this case you have to list the parameter *after* the functions
that you want to refer to.


State machine
-------------

A formally defined finite state machine is necessary to ensure and reason about
correct behaviour. Concert provides an implicitly defined, decorator-based state
machine. All you need to do is declare a :class:`.State` object on the base
device class and apply the :func:`.transition` decorator on each method that
changes the state of a device::

    from concert.fsm import State, transition

    class Motor(Device):

        state = State(default='open')

        ...

        @state.transition(source='standby', target='moving')
        def start_moving(self):
            ...

If the source state is valid on such a device, ``start_moving`` will run and
eventually change the state to ``moving``. In case of two-step functions, an
``immediate`` state can be set that is valid throughout the body of the
function::

        @state.transition(source='standby', target='standby', immediate='moving')
        def move(self):
            ...

Besides single state strings you can also add arrays of strings and a catch-all
``*`` state that matches all states.

If an exceptional behaviour happens during the execution the device is put
automatically into an error state::

        @state.transition(source='*')
        def move(self):
            ...
            if cannot_move:
                raise Exception("Uh, something bad happened")


Parameters
~~~~~~~~~~

In case changing a parameter value causes a state transition, you can list
the source and target states in the :class:`.Parameter` object::

    class Motor(Device):

        state = State(default='standby')

        velocity = Parameter(unit=q.m / q.s,
                             source='*',
                             target='moving')
