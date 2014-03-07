from concert.base import Selection
from concert.devices.detectors import base


class Detector(base.Detector):

    scintillator = Selection(list(range(1, 12)), help="Current scintillator")
    use = Selection([1, 2, 3], help="Current camera in use")

    def __init__(self, cameras):
        if len(cameras) < 3:
            raise ValueError("You must pass at least three cameras")

        super(Detector, self).__init__(cameras[0], 2.0)

        self._cameras = cameras
        self._current_camera = 1
        self._current_scintillator = 1

    def _set_use(self, value):
        self._current_camera = value
        self._camera = self._cameras[value - 1]

    def _get_use(self):
        return self._current_camera

    def _set_scintillator(self, value):
        self._current_scintillator = value

    def _get_scintillator(self):
        return self._current_scintillator
