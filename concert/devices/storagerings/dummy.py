"""
Dummy ring to test asychronous updates.
"""
import time
import threading
from concert.quantities import q
from concert.devices.storagerings import base


class StorageRing(base.StorageRing):

    """A storage ring dummy."""

    def __init__(self):
        super(StorageRing, self).__init__()
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

        self.monitor = threading.Thread(target=update)
        self.monitor.daemon = True
        self.monitor.start()

    def _get_current(self):
        return self._current - self._lifetime * self._current_decay

    def _get_energy(self):
        return self._energy - self._lifetime * self._energy_decay

    def _get_lifetime(self):
        return self._lifetime
