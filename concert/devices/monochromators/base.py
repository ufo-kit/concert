'''
Created on Apr 11, 2013

@author: farago
'''
from concert.base import Device, Parameter
import quantities as q


class Monochromator(Device):
    """Monochromator device which is used to filter the beam in order to 
    get a very narrow energy bandwidth.
    """
    def __init__(self, calibration, limiter=None):
        params = [Parameter("energy", self._get_energy, self._set_energy,
                            q.eV, limiter, "Monochromatic energy")]
        super(Monochromator, self).__init__(params)
        self._calibration = calibration
        
    def _get_calibrated_energy(self):
        return self._calibration.to_user(self._get_energy())
    
    def _set_calibrated_energy(self, energy):
        self._set_energy(self._calibration.to_steps(energy))
        
    def _get_energy(self):
        raise NotImplementedError
    
    def _set_energy(self, steps):
        raise NotImplementedError