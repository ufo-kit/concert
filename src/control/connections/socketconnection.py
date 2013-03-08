'''
Created on Mar 5, 2013

@author: farago
'''
import socket
import logging
from control.connections.connection import Connection


class SocketConnection(Connection):
    def __init__(self, uri, host, port):
        super(SocketConnection, self).__init__(str(host) + ":" + str(port))
        self._peer = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(20)
        self._sock.connect(self._peer)
        self._logger = logging.getLogger(__name__ + "." +\
                                         self.__class__.__name__)

    def __del__(self):
        self._sock.close()

    def communicate(self, cmd, *args):
        """Both-way communication via a socket.

        @param cmd: command to send
        @param data: data to send

        @return: reponse from the device

        """
        to_send = cmd % (args)
        self._logger.debug('Sending {0}'.format(to_send))
        self._sock.sendall(to_send.encode('ascii'))

        try:
            result = self._sock.recv(1024)
            self._logger.debug('Received {0}'.format(result))
            return result
        except socket.timeout:
            self._logger.warning('Reading from %s:%i timed out' % self._peer)
