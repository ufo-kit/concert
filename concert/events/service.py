'''
Created on Mar 15, 2013

@author: farago
'''
from concert.events.dispatcher import dispatcher


def wait(events, timeout=None):
    """Wait for *events*. If *timeout* is given, then every event will be
    given a *timeout* time."""
    dispatcher.wait(events, timeout)
