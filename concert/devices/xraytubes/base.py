"""
An X-ray tube.
"""
from concert.async import async
from concert.base import check
from concert.quantities import q
from concert.base import Quantity, AccessorNotImplementedError, State
from concert.devices.base import Device


class XRayTube(Device):
    """
    A base x-ray tube class.
    """

    voltage = Quantity(q.kV)
    current = Quantity(q.uA)
    power = Quantity(q.W)

    state = State(default='off')

    def __init__(self):
        super(XRayTube, self).__init__()

    def _get_state(self):
        raise AccessorNotImplementedError

    def _get_voltage(self):
        raise AccessorNotImplementedError

    def _set_voltage(self, voltage):
        raise AccessorNotImplementedError

    def _get_current(self):
        raise AccessorNotImplementedError

    def _set_current(self, current):
        raise AccessorNotImplementedError

    def _get_power(self):
        return (self.voltage*self.current).to(q.W)

    @async
    @check(source='off', target='on')
    def on(self):
        """
        on()

        Enables the x-ray tube.
        """
        self._on()

    @async
    @check(source='on', target='off')
    def off(self):
        """
        off()

        Disables the x-ray tube.
        """
        self._off()

    def _on(self):
        """
        Implementation of on().
        """
        raise NotImplementedError

    def _off(self):
        """
        Implementation of off().
        """
        raise NotImplementedError
