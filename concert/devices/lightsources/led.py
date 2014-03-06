"""
LEDs
"""
from concert.quantities import q
from concert.base import Quantity
from concert.devices.io.wago import IO
from concert.devices.lightsources import base


class Led(base.LightSource): 
    """
    The class for control of LED's intensity by voltage 
    """       
    voltage = Quantity(q.V, lower=0 * q.V, upper=30 * q.V)    
    
    def __init__(self, host, port, address):
        self._host = host
        self._port = port
        super(Led, self).__init__()
        
    def _set_voltage(self, port, value):        
        """
        Set voltage for the LED
        """        
        connection = IO(self._host, self._port)
        connection._write_port(port, value.magnitude * 1092)
            
    def _get_voltage(self, port):
        """
        Get current voltage from the LED
        """
        connection = IO(self._host, self._port)
        voltage = float(connection._read_port(port))
        return voltage * q.V / 1092
