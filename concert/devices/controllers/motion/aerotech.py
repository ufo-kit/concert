'''
Created on Apr 11, 2013

@author: farago
'''
from concert.devices.base import Device
from concert.devices.motors.aerotech import Aerorot
from concert.connections.inet import Aerotech, Connection
from concert.asynchronous import async
import logbook


class HLe(Device):
    """Aerotech Ensemble HLe controller."""
    HOST = "192.168.18.19"
    PORT = 8001

    def __init__(self):
        self._connection = Aerotech(HLe.HOST, HLe.PORT)
        self.logger = logbook.Logger(self.__class__, __name__)
        super(HLe, self).__init__()

    def reset(self):
        """Reset the controller."""
        # TODO: timeout here?
        linked = False
        self._connection.execute("RESET")

        while not linked:
            try:
                self._connection = Aerotech(HLe.HOST, HLe.PORT)
            except:
                pass
            else:
                linked = True

    @async
    def get_positions(self):
        self.program_run(1, "position_readout.bcx")
        positions = []

        conn = Aerotech(HLe.HOST, 8000)

        while True:
            data = conn.execute("NEXT")
            if data == "ERROR":
                msg = "Error reading positions."
                self.logger.error(msg)
                raise RuntimeError(msg)
            elif data == "EOF":
                break
            positions.append(float(data))

        return positions

    def program_run(self, task, program_name):
        """Execute *program_name* on task number *task*."""
        self._connection.execute("TASKRUN %d, \"%s\"" %
                                 (task, program_name))

    def program_stop(self, task):
        """Stop program execution on task number *task*."""
        self._connection.execute("PROGRAM STOP %d" % (task))

    def get_integer_register(self, register):
        """Get value stored in integer *register* on the controller."""
        self._connection.execute("IGLOBAL(%d)" % (register))

    def set_integer_register(self, register, value):
        """Set *value* stored in integer *register* on the controller."""
        self._connection.execute("IGLOBAL(%d)=%f" % (register, value))

    def get_double_register(self, register):
        """Get value stored in double *register* on the controller."""
        self._connection.execute("DGLOBAL(%d)" % (register))

    def set_double_register(self, register, value):
        """Set *value* stored in double *register* on the controller."""
        self._connection.execute("DGLOBAL(%d)=%f" % (register, value))
