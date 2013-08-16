"""Aerotech"""
import time
from concert.quantities import q
from concert.connections.inet import Aerotech
from concert.devices.base import LinearCalibration
from concert.devices.motors.base import ContinuousMotor, Motor


class Aerorot(ContinuousMotor):

    """Aerorot (Continuous Motor) class implementation."""
    HOST = "192.168.18.19"
    PORT = 8001
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

    def __init__(self):
        pos_calib = LinearCalibration(1 / q.deg, 0 * q.deg)
        velo_calib = LinearCalibration(1 / (q.deg / q.sec), 0 * q.deg / q.sec)
        super(Aerorot, self).__init__(pos_calib, velo_calib)

        self["position"].unit = q.deg
        self["velocity"].unit = q.deg / q.sec

        self._connection = Aerotech(Aerorot.HOST, Aerorot.PORT)
        self._connection.execute("ENABLE %s" % (Aerorot.AXIS))

    def __del__(self):
        self._connection.execute("DISABLE %s" % (Aerorot.AXIS))

    def _query_state(self):
        return int(self._connection.execute("AXISSTATUS(%s)" % (Aerorot.AXIS)))

    def _get_position(self):
        return float(self._connection.execute("PFBK(%s)" % (Aerorot.AXIS)))

    def _set_position(self, steps):
        self._connection.execute("MOVEABS %s %f" % (Aerorot.AXIS, steps))

        while not self._query_state() >> Aerorot.AXISSTATUS_IN_POSITION & 1:
            time.sleep(Aerorot.SLEEP_TIME)

    def _get_velocity(self):
        return float(self._connection.execute("VFBK(%s)" % (Aerorot.AXIS)))

    def _set_velocity(self, steps):
        self._connection.execute("FREERUN %s %f" % (Aerorot.AXIS, steps))

        while self._query_state() >> Aerorot.AXISSTATUS_ACCEL_PHASE & 1:
            time.sleep(Aerorot.SLEEP_TIME)

    def _get_state(self):
        res = self._query_state()
        if res >> Aerorot.AXISSTATUS_MOVE_ACTIVE & 1:
            state = Motor.MOVING
        elif res >> Aerorot.AXISSTATUS_IN_POSITION & 1:
            state = Motor.STANDBY
        else:
            state = Motor.NA

        return state

    def _stop(self):
        if self.state == Motor.MOVING:
            self._connection.execute("ABORT %s" % (Aerorot.AXIS))

        while self.state == Motor.MOVING:
            time.sleep(Aerorot.SLEEP_TIME)

    def _home(self):
        self._connection.execute("HOME %s" % (Aerorot.AXIS))

        while self.state == Motor.MOVING:
            time.sleep(Aerorot.SLEEP_TIME)

    def get_digital_in(self, port, bit):
        """Get TTL level on *port* and *bit*."""
        res = self._connection.execute("DIN(%s,%d,%d)" %
                                       (Aerorot.AXIS, port, bit))
        return res == "1"

    def set_digital_out(self, port, bit, value):
        """Set TTL level on *port* and *bit* to *value*"""
        self._connection.execute("DOUT %s, %d, %d:%d" % (Aerorot.AXIS, port,
                                                         bit, value))
