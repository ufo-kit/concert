'''
Created on Mar 4, 2013

@author: farago
'''
from device.motion.axis.axis import TangoDiscreteAxis

class ANKATANGODiscreteAxis(TangoDiscreteAxis):
    def _is_hard_position_limit_reached(self):
        return self._tango_device.BackwardLimitSwitch or\
                self._tango_device.ForwardLimitSwitch
                
    def _get_position_real(self):
        return self._tango_device.position
                
    def _set_position_real(self, position):
        self._tango_motor.write_attribute("position", position)
        
if __name__ == '__main__':
    pass