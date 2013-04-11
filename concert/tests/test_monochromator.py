'''
Created on Apr 11, 2013

@author: farago
'''
import unittest
from concert.devices.motors.base import LinearCalibration
import quantities as q
from concert.devices.monochromators.dummy import DummyMonochromator


class TestDummyMonochromator(unittest.TestCase):
    def setUp(self):
        calibration = LinearCalibration(1*q.eV, 0*q.eV)
        self.mono = DummyMonochromator(calibration)