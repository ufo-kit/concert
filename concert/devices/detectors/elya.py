from concert.devices.detectors import base


class Detector(base.Detector):

    camera = Selection([1, 2, 3])
    magnification = Range(1.0, 4.0)
    filter = Selection([1, 2, 3, 4])

    def __init__(self, cameras=None):
        super(Detector, self).__init__()

        # These are concert.devices.cameras.* objects
        self._cameras = cameras

    def current(self):
        """Return currently selected camera object."""

        # This could (?) come in pretty handy as in
        #   
        #   with detector.current().recording():
        #       frames = acquire(detector.current())
        #
        return self._cameras(self.camera.index - 1)

    def _set_camera(self, value):
        # rotate the dish washer
        pass

    def _get_camera(self):
        pass

    def _set_magnification(self, value):
        # adjust the lens
        pass

    def _get_magnification(self):
        pass

    def _set_filter(self, value):
        pass

    def _get_filter(self, value):
        pass
