'''
Created on Mar 5, 2013

@author: farago
'''
from control.connections.connection import Connection
import PyTango
import os


class TangoConnection(Connection):
    def __init__(self, uri, tango_host=None, tango_port=None):
        super(TangoConnection, self).__init__(uri)
        # Set the host and port for connecting to the Tango database.
        # TODO: check if there is a way to adjust the host in PyTango.
        if tango_host is not None and tango_port is not None:
            os.environ["TANGO_HOST"] = "%s:%d" % (tango_host, tango_port)
            
        self._tango_device = PyTango.DeviceProxy(self._uri)

    @property
    def tango_device(self):
        return self._tango_device