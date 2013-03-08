'''
Created on Mar 5, 2013

@author: farago
'''
from control.connections.connection import Connection


class DummyConnection(Connection):
    def __init__(self):
        Connection.__init__(self, "dummy_uri")
