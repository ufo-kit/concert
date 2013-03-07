'''
Created on Mar 4, 2013

@author: farago
'''


class Connection(object):
    def __init__(self, uri):
        self._uri = uri
        
    @property
    def uri(self):
        return self._uri

    def communicate(self, cmd, *args):
        raise NotImplementedError