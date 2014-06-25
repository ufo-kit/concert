"""Aerotech"""
from concert.helpers import busy_wait
from concert.quantities import q
from concert.networking.aerotech import Connection
from concert.devices.motors.base import ContinuousRotationMotor


class Aerorot(ContinuousRotationMotor):

    """Aerorot ContinuousRotationMotor class implementation."""
    AXIS = "X"

    # status constants (bits of the AXISSTATUS output (see HLe docs))
    AXISSTATUS_ENABLED = 0
    AXISSTATUS_HOMED = 1
    AXISSTATUS_IN_POSITION = 2
    AXISSTATUS_MOVE_ACTIVE = 3
    AXISSTATUS_ACCEL_PHASE = 4
    AXISSTATUS_DECEL_PHASE = 5
    AXISSTATUS_POSITION_CAPTURE = 6
    AXISSTATUS_HOMING = 14

    SLEEP_TIME = 0.01

    def __init__(self, host, port=8001, enable=True):
        super(Aerorot, self).__init__()

        self._connection = Connection(host, port)
        if enable:
            self.enable()

    def enable(self):
        """Enable the motor."""
        self._connection.execute("ENABLE %s" % (Aerorot.AXIS))

    def disable(self):
        """Disable the motor."""
        self._connection.execute("DISABLE %s" % (Aerorot.AXIS))

    def _query_state(self):
        return int(self._connection.execute("AXISSTATUS(%s)" % (Aerorot.AXIS)))

    def _get_position(self):
        return float(self._connection.execute("PFBK(%s)" % (Aerorot.AXIS))) * q.deg

    def _set_position(self, position):
        self._connection.execute("MOVEABS %s %f" % (Aerorot.AXIS, position.magnitude))

        # If this is not precise enough one can try the IN_POSITION
        self['state'].wait('standby', sleep_time=Aerorot.SLEEP_TIME)

    def _get_velocity(self):
        return float(self._connection.execute("VFBK(%s)" % (Aerorot.AXIS))) * q.deg / q.s

    def _set_velocity(self, velocity):
        self._connection.execute("FREERUN %s %f" % (Aerorot.AXIS, velocity.magnitude))

        busy_wait(self._is_velocity_stable, sleep_time=Aerorot.SLEEP_TIME)

    def _get_state(self):
        res = self._query_state()

        # Simplified behavior because of unstable motor states, i.e.
        # error after long-homing etx. TODO:investigate
        state = 'standby'

        if res >> Aerorot.AXISSTATUS_MOVE_ACTIVE & 1:
            state = 'moving'

        return state

    def _stop(self):
        if self.check_state() == 'moving':
            self._connection.execute("ABORT %s" % (Aerorot.AXIS))

        self['state'].wait('standby', sleep_time=Aerorot.SLEEP_TIME)

    def _home(self):
        self._connection.execute("HOME %s" % (Aerorot.AXIS))

        self['state'].wait('standby', sleep_time=Aerorot.SLEEP_TIME)

    def _is_velocity_stable(self):
        accel = self._query_state() >> Aerorot.AXISSTATUS_ACCEL_PHASE & 1
        decel = self._query_state() >> Aerorot.AXISSTATUS_DECEL_PHASE & 1

        return not (accel or decel)
