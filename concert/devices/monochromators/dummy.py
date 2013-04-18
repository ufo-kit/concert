'''
Created on Apr 11, 2013

@author: farago
'''
from concert.devices.monochromators import base


class Monochromator(base.Monochromator):
    def __init__(self, calibration):
        super(Monochromator, self).__init__(calibration)
        self._energy = None

    def _get_energy(self):
        return self._energy

    def _set_energy(self, energy):
        self._energy = energy
