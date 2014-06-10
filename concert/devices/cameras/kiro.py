from concert.devices.cameras import base
from concert.devices.cameras import uca


class Camera(base.Camera):

    def __init__(self, ip, port):
        super(Camera, self).__init__()

        self._tango_camera = None
        # fetch properties from remote camera

        self._uca_camera = uca.Camera('kiro')
        self._uca_camera.set_properties(ip=ip, port=port)

    def _record_real(self):
        pass    # use tango

    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass

    def _grab_real(self):
        return self._uca_camera.grab()
