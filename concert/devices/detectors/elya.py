from concert.quantities import q
from concert.base import Selection, Quantity
from concert.async import async
from concert.devices.base import Device
from concert.devices.detectors import base


class HighSpeedLightpath(Device):

    magnification = Selection([1.0, 2.0, 3.0, 4.0] * q.dimensionless,
                              help="Discrete magnification")

    aperture = Selection([1.4, 2.0, 2.8, 4.0, 5.6],
                         help="aperture in f-number")

    def __init__(self):
        super(HighSpeedLightpath, self).__init__()

    def _get_magnification(self):
        return 2.0 * q.dimensionless

    def _set_magnification(self, value):
        pass

    def _get_aperture(self):
        return 1.4

    def _set_aperture(self, value):
        pass


class HighResolutionLightpath(Device):

    magnification = Quantity(unit=q.dimensionless,
                             lower=1.0 * q.dimensionless,
                             upper=16.0 * q.dimensionless,
                             help="Continuous magnification")

    filter = Selection(list(range(1, 12)))

    def __init__(self):
        super(HighResolutionLightpath, self).__init__()

    def _get_magnification(self):
        return 3.3 * q.dimensionless

    def _set_magnification(self, value):
        pass


class Detector(base.Detector):

    scintillator = Selection(list(range(1, 12)),
                             help="Current scintillator")

    use = Selection([1, 2, 3],
                    help="Current camera in use")

    def __init__(self, cameras):
        if len(cameras) < 3:
            raise ValueError("You must pass at least three cameras")

        super(Detector, self).__init__(cameras[0], 2.0)

        self._cameras = cameras
        self._highspeed_path = HighSpeedLightpath()
        self._highres_path = HighResolutionLightpath()

        self._current_camera = 1
        self._current_scintillator = 1
        self._current_lightpath = self._highspeed_path

    def _set_use(self, value):
        self._current_camera = value
        self._camera = self._cameras[value - 1]

    def _get_use(self):
        return self._current_camera

    def _set_scintillator(self, value):
        self._current_scintillator = value

    def _get_scintillator(self):
        return self._current_scintillator

    @async
    def use_fast_mode(self):
        self._current_lightpath = self._highspeed_path

    @async
    def use_high_resolution_mode(self):
        self._current_lightpath = self._highres_path

    @property
    def lightpath(self):
        return self._current_lightpath

    @property
    def magnification(self):
        return self.lightpath.magnification

    @magnification.setter
    def magnification(self, value):
        self.lightpath.magnification = value
