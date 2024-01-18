===============================
Writing devices and experiments
===============================

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
module and `pytest`_. To run all tests, you can call pytest directly in the root
directory or run make with the ``check`` argument ::

    $ make check

Some tests take a lot of time to complete and are marked with the ``@slow``
decorator. To skip them during regular development cycles, you can run ::

    $ make check-fast

You are highly encouraged to add new tests when you are adding a new feature to
the core or fixing a known bug.

.. _pytest: https://docs.pytest.org/



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


Asynchronous constructors
=========================

Devices and many other classes in concert subclass
:class:`concert.base.AsyncObject` which does not use the classical ``def
__init__(...)`` constructor but an ``async def __ainit__(...)``. That is because
parameter getters and setters are coroutine funcions (``async def``) and when a
Parameterizable instance is created, there is a good chance that some parameters
should be read or written and that must be done with the ``await param.get()``
syntax and that is only possible in coroutine functions, which a normal
``__init__`` constructor is not. Hence, we introduced a new kind of constructor
``__ainit__`` which allows such syntax. Inheritance works as usual but if your
class inherits from another :class:`.AsyncObject` (the base of Parameterizable)
*and* a normal class with just an ``__init__`` contructor, you need to call
*both* in your constructor, like this::

    class Foo(Parameterizable, StandardClass):
        async def __ainit__(self, async_param, sync_param):
            await super().__ainit__(async_param)
            super().__init__(sync_param)

Classes subclassing :class:`.AsyncObject` cannot define ``__init__``
constructors, which would lead to ambiguities.


Adding a new device
===================

To add a new device to an existing device class (such as motor, pump,
monochromator etc.), a new module has to be added to the corresponding device
class package. Inside the new module, the concrete device class must then import
the base class, inherit from it and implement all abstract method stubs.

Concert is based on `asyncio`_, see also the :ref:`user documentation
<concurrent-execution>`. In order for the concurrent execution to work well, all
concert code needs to adhere to the concepts of asyncio and the device
implementations as well. That means that all methods which actually manipulate
the device in any way need to be defined as *async def*. All parameter getters
and setters already are defined in this way, and so have to be their underscored
implementations (see below).

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

        async def _get_position(self):
            # the returned value must have units compatible with units set in
            # the Quantity this getter implements. In this case we just return
            # some stored value
            return self._read_position

        async def _set_position(self, position):
            # position is guaranteed to be in the units set by the respective
            # Quantity. In this case just store the desired position in a
            # private variable.
            self._read_position = position

We guarantee that setters which implement a :class:`.Quantity`, like the
:meth:`._set_position` above, obtain the value in the exact same units as they
were specified in the respective :class:`.Quantity` they implement. E.g. if the
above :meth:`_set_position` implemented a quantity with units set in kilometers,
the :attr:`~.LinearMotor.position` of the :meth:`._set_position` will also be in
kilometers.  On the other hand the getters do not need to return the exact same
quantity but the value must be compatible, so the above :meth:`._get_position`
could return millimeters and the user would get the value in kilometers, as
defined in the respective :class:`.Quantity`.

Parameter setters can be cancelled by hitting *ctrl-c* or *ctrl-k*. If you want
a parameter to make some cleanup action after *ctrl-c* is pressed, you should
catch the ``asyncio.CancelledError`` exception, for the motor above you can
write::

        async def _set_position(self, position):
            try:
                self._read_position = position
            except asyncio.CancelledError:
                # cleanup action goes here
                raise   # re-raise the exception if needed


And you are guaranteed that when you interrupt the setter the motor stops
moving.

.. _asyncio: https://docs.python.org/3/library/asyncio.html


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

        async def __ainit__(self):
            await super(Pump, self).__ainit__()

The `flow_rate` parameter can only receive values from zero to one cubic meter
per second.

We didn't specify explicit *fget* and *fset* functions, which is why  implicit
setters and getters called `_set_flow_rate` and `_get_flow_rate` are installed.
The real devices then need to implement these. You can however, also specify
explicit setters and getters in order to hook into the get and set process::

    class Pump(Device):

        async def __ainit__(self):
            await super(Pump, self).__ainit__()

        async def _intercept_get_flow_rate(self):
            return await self._get_flow_rate() * 10

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

        async def _set_flow_rate(self, flow_rate):
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
        async def start(self):
            ...

        async def _start(self):
            # the actual implementation of starting something
            ...


    class Motor(BaseMotor):

        """A motor with hardware state reading support."""

        ...

        async def _start(self):
            # Implementation communicates with hardware
            ...

        async def _get_state(self):
            # Get the state from the hardware
            ...


    class StatelessMotor(BaseMotor):

        """A motor which doesn't support state reading from hardware."""

        # we have to specify a default value since we cannot get it from
        # hardware
        state = State(default='standby')

        ...

        @transition(target='moving')
        async def _start(self):
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


Limits
~~~~~~

:class:`.Quantity` instances can have user-defined or external limits (e.g. read
from a controller). There are :attr:`.Quantity.lower` and
:attr:`.Quantity.upper` limits and they are obtained in the following way. If
:func:`external_lower_getter` function is specified in the constructor of the
quantity, it is used to get the lower limit. If it is not, then the user-defined
limit is returned, and that is done either via the :func:`user_lower_getter`
function if specified in the constructor of the quantity, or via the value
saved in the quantity, set previousy by :meth:`.QuantityValue.set_lower`. The setter
calls the :func:`user_lower_setter` if specified, otherwise just saves the value
in a variable inside the quantity. The user-defined getters and setters are
useful for invoking mechanisms beyond concert, e.g. updating the limits in a
Tango database. The limits can be locked in a similar way to parameter locking.


Acquisitions
============

Each experiment consist of a set of :class:`.LocalAcquisition` or
:class:`.RemoteAcquisition` instances which generate images. The purpose of the
acquisition class is to trigger the data acquisition and connect the processing
consumers to it.

In case of :class:`.LocalAcquisition`, the image data is is produced by an async
generator and forwarded to :class:`.LocalConsumer` instances. The splitting of
the data stream is handled by the acquisition. Local consumers wrap data
processing coroutine functions which do the actual processing.

In case of :class:`.RemoteAcquisition`, the images must be streamed via `ZMQ
<https://pyzmq.readthedocs.io/en/latest/>`_ by the producer and the consumers must be of type
:class:`.RemoteConsumer`. In this case, the producer may be an async generator
yielding arbitrary values or a coroutine function returning the amount of
produced images. Either way, the number of generated
images is used in the call to :meth:`.RemoteConsumer.wait`, which must block
until the remote processing has finished.


Creating a experiment class
===========================

A new Experiment inherits from :class:`.Experiment`.
Like the :class:`.Device` an experiment class can also hold :class:`.Quantity` and :class:`.Parameter`.
The logger from the :class:`.Experiment` will automatically write the values of these in the experiments log file.
It also has a state parameter, showing the current experiments state.

An example experiment with one :class:`.LocalAcquisition` can look like this::

    class MyExperiment(Experiment):
        num_images = Parameter(help="number of images to acquire")

        async def __ainit__(self, camera, walker):
            self._num_images = 5
            self._camera = camera
            image_acquisition = Acquisition("images", self._acquire_images)
            await super().__init__(acquisitions=[image_acquisition], walker=walker)

        async def _get_num_images(self):
            return self._num_images

        async def _set_num_images(self, n):
            self._num_images = int(n)

        async def _acquire_images(self):
            await self._camera.set_trigger_source("AUTO")
            async with self._camera.recording():
                for i in range(await self.get_num_images()):
                    yield await self._camera.grab()
