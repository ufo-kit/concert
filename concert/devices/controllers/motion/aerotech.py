'''
Created on Apr 11, 2013

@author: farago
'''
from concert.base import Device
from concert.connection import AerotechConnection
from concert.devices.motors.aerotech import Aerorot


class HLe(Device):
    """Aerotech Ensemble HLe controller."""
    HOST = ""
    PORT = 0
    
    def __init__(self):
        self._connection = AerotechConnection(HLe.HOST, HLe.PORT)
        self._motors = [Aerorot()]
        
    def _get_motors(self):
        return self._motors
    
    def reset(self):
        """Reset the controller."""
        # TODO: timeout here?
        linked = False
        self._connection.execute("RESET")
        
        while not linked:
            try:
                self._connection = AerotechConnection(HLe.HOST, HLe.PORT)
            except:
                pass
            else:
                linked = True
        
    def program_run(self, task, program_name):
        """Execute *program_name* on task number *task*."""
        self._connection.execute("PROGRAM RUN %d, \"%s\"" %
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
