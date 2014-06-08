from concert.devices.cameras import base
from concert.devices.cameras import uca


class Camera(base.Camera):

    def __init__(self, ip=None, port=None):
        super(Camera, self).__init__()

        self._uca_camera = uca.Camera('kiro')
        self._tango_camera = None

        if ip and port:
            self.connect(ip, port)

    def connect(self, ip, port):
        self._uca_camera.set_properties(ip=ip, port=port)
        # Here we also need to get the properties from the tango device and map
        # them to device properties.

    def _record_real(self):
        pass    # use tango

    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass

    def _grab_real(self):
        return self._uca_camera.grab()
