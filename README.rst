Concert is a light-weight control system interface to control Tango and native
devices like this::

    import quantities as q
    from concert.devices.motors.crio import LinearMotor

    motor = LinearMotor()
    motor.set_position(10 * q.mm)
    motor.move(-5 * q.mm)
