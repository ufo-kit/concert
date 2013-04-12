'''
Created on Apr 12, 2013

@author: farago
'''
import unittest
from concert.devices.shutters.dummy import DummyShutter


class TestDummyShutter(unittest.TestCase):
    def setUp(self):
        self.shutter = DummyShutter()

    def test_open(self):
        self.shutter.open().wait()
        self.assertTrue(self.shutter.is_open())

    def test_close(self):
        self.shutter.close().wait()
        self.assertFalse(self.shutter.is_open())
