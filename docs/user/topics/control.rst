.. _controlling-devices:

==============
Device control
==============

Parameters
==========

In Concert, a *device* is a software abstraction for a piece of hardware that
can be controlled. Each device consists of a set of named :class:`Parameter`
instances and device-specific methods. If you know the parameter name, you can
get a reference to the parameter object by using the index operator::

    pos_parameter = motor['position']

To set and get parameters explicitly , you can use the :meth:`Parameter.get`
and :meth:`Parameter.set` coroutine methods::

    await pos_parameter.set(1 * q.mm)
    print (await pos_parameter.get())

Both methods return a coroutine (see :ref:`concurrent-execution` for details)
and give the control back to you, so that other things can (and should) happen
concurrently. As you can see, to get the result of a coroutine you use the
``await`` keyword.

An easier way to set and get parameter values are properties via the
dot-name-notation::

    motor.position = 1 * q.mm
    print (motor.position)

As you can see, accessing parameters this way will *always be synchronous* and
*block* execution until the value is set or fetched. If you press *ctrl-c* while
you are setting a parameter the function will stop and a cancelling action will
be called, like stopping a motor, so that you don't accidentaly crush your
devices. However, please be aware that this is up to device implementation, so
you should check if the device you are using is safe in this manner.

Parameter objects are not only used to communicate with a device but also carry
meta data information about the parameter. The most important ones are
:attr:`Parameter.name`, :attr:`Quantity.unit` (in case of a parameter having a
physical unit, like motor's position) and the doc string describing the
parameter. Moreover, parameters can be queried for access rights using
:attr:`Parameter.writable`.

To get all parameters of an object, you can iterate over the device itself ::

    for param in motor:
        print("{0} => {1}".format(param.unit if hasattr(param,'unit') else None, param.name))


Saving state
------------

In some scenarios you would like to come back to a certain state. Let's suppose,
you have a motor that you want to check if it moves. If it does, you want it to
go back to the same place it came from. For these cases you can use
:meth:`Device.stash` to store the current state of a device and
:meth:`Device.restore` to go back. Because this is done in a stacked fashion,
you can, for example, model local coordinate pretty easily::

   await motor.stash()

   # Do movements aka modify the "local" coordinate system
   await motor.move(1 * q.mm)

   # Go back to the original state
   await motor.restore()


Locking parameters
------------------

In case you want to prevent a parameter from being written you can use
:meth:`.ParameterValue.lock`. If you specify a *permanent* parameter to be True
the parameter cannot be unlocked anymore. In case you want to unlock
a parameter you can use :meth:`.ParameterValue.unlock`, to get the state
you can check the attribute :attr:`.ParameterValue.locked`. All the
parameters within a device can be locked and unlocked at once, for example
one can do::

    motor['position'].lock()
    motor.position = 10 * q.mm
    # Does not work, you will get a LockError
    motor['position'].locked
    True

    motor['position'].unlock()

    # Works as expected
    motor.position = 10 * q.mm

    # Lock the whole device (all parameters)
    motor.lock(permanent=True)

    # This will not work anymore
    motor.unlock()
    # You will get a LockError



Emergency stop
--------------

On *ctrl-k*, the background tasks are cancelled and on top of that on all
devices :meth:`.Device.emergency_stop` will be called in order to bring them to
a standstill.
