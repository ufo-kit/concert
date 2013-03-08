'''
Created on Mar 3, 2013

@author: farago
'''
import itertools


make_event_id = itertools.count().next


class StateChangeEvent(object):
    STATE = make_event_id()
