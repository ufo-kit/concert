'''
Created on Mar 3, 2013

@author: farago
'''
import itertools

next_id = itertools.count().next
    
class State(object):
    ERROR = next_id()