Concert is a light-weight control system interface to control Tango and native
devices like this::

    import quantities as q
    from concert.devices.axis.crio import LinearAxis

    axis = LinearAxis()
    axis.set_position(10 * q.mm)
    axis.move_by(-5 * q.mm)
