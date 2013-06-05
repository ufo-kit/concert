"""
Dummy ring to test asychronous updates.
"""
from concert.devices.storagerings.base import StorageRing
import quantities as q
import time
import threading


class DummyRing(StorageRing):
    """Create a Dummy Ring."""
    def __init__(self):
        super(DummyRing, self).__init__()
        self._lifetime = 10 * q.hour
        self._current = 100 * q.mA
        self._energy = 5 * q.MeV
        self._current_decay = 0.05 * q.mA / q.hour
        self._energy_decay = 0.05 * q.MeV / q.hour

        def update():
            """Test updates."""
            while True:
                # Yay, we invented a time machine \o/
                self._lifetime += 1 * q.hour
                time.sleep(5.0)

                # We want to tell everyone, that the parameters are updated.
                # Now we abuse the fact that the parameters are owner_only and
                # the setter doesn't do anything.
                self['energy'].notify()
                self['current'].notify()

        self.monitor = threading.Thread(target=update)
        self.monitor.daemon = True
        self.monitor.start()

    def _get_current(self):
        return self._current - self._lifetime * self._current_decay

    def _get_energy(self):
        return self._energy - self._lifetime * self._energy_decay

    def _get_lifetime(self):
        return self._lifetime
