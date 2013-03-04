'''
Created on Mar 3, 2013

@author: farago
'''
import itertools

get_type_id = itertools.count().next

class Motion(object):
    START = get_type_id()
    STOP = get_type_id()
    LIMIT_BREACH = get_type_id()
    
class ContinuousMotion(Motion):
    VELOCITY_STEADY = get_type_id()