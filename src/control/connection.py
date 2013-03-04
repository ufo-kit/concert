'''
Created on Mar 4, 2013

@author: farago
'''
import os
import PyTango
import socket
import logging


class _Connection(object):
    def __init__(self, uri):
        self._uri = uri

class TangoConnection(_Connection):
    def __init__(self, uri, tango_host=None, tango_port=None):
        super(TangoConnection, self).__init__(uri)
        # Set the host and port for connecting to the Tango database.
        # TODO: check if there is a way to adjust the host in PyTango.
        if tango_host is not None and tango_port is not None:
            os.environ["TANGO_HOST"] = "%s:%d" % (tango_host, tango_port)
            
        self._tango_device = PyTango.DeviceProxy(self._uri)
        
class SocketConnection(object):
    def __init__(self, uri, host, port):
        self._peer = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(20)
        self._sock.connect(self._peer)
        self._logger = logging.getLogger(__name__ + "." +\
                                         self.__class__.__name__)

    def __del__(self):
        self._sock.close()

    def send(self, data):
        self._logger.debug('Sending {0}'.format(data))
        self._sock.sendall(data.encode('ascii'))

        try:
            result = self._sock.recv(1024)
            self._logger.debug('Received {0}'.format(result))
        except socket.timeout:
            self._logger.warning('Reading from %s:%i timed out' % self._peer)