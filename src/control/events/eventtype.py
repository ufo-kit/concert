'''
Created on Mar 3, 2013

@author: farago
'''
import itertools

get_type_id = itertools.count().next

class Motion(object):
    START = get_type_id()
    STOP = get_type_id()
    STATE_CHANGED = get_type_id()
    POSITION_CHANGED = get_type_id()
    VELOCITY_CHANGED = get_type_id()
    LIMIT_REACHED = get_type_id()