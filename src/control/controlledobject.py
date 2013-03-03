'''
Created on Mar 3, 2013

@author: farago
'''
import uuid


class Identifiable(object):
    """
    An object's id unique for the duration of a process in which the object
    resides. Python's id() function does not guarantee unique object ids
    in terms of process lifetime, only in terms of object lifetime.
    """
    def __init__(self):
        self._id = uuid.uuid4()
        
    @property
    def object_id(self):
        return self._id