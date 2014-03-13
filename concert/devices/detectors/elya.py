from concert.quantities import q
from concert.base import Selection, Quantity
from concert.async import async
from concert.helpers import Bunch
from concert.devices.base import Device
from concert.devices.detectors import base


class HighSpeedLightpath(Device):

    magnification = Quantity(unit=q.dimensionless,
                             lower=1.0 * q.dimensionless,
                             upper=16.0 * q.dimensionless,
                             help="Continuous magnification")

    aperture = Selection([1.4, 2.0, 2.8, 4.0, 5.6],
                         help="aperture in f-number")

    def __init__(self):
        super(HighSpeedLightpath, self).__init__()

    def _get_magnification(self):
        return 3.3 * q.dimensionless

    def _set_magnification(self, value):
        pass

    def _get_aperture(self):
        return 1.4

    def _set_aperture(self, value):
        pass


class HighResolutionLightpath(Device):

    magnification = Selection([1.0, 2.0, 3.0, 4.0] * q.dimensionless,
                              help="Discrete magnification")

    filter1 = Selection(list(range(1, 6)))
    filter2 = Selection(list(range(1, 6)))

    def __init__(self):
        super(HighResolutionLightpath, self).__init__()

    def _get_magnification(self):
        return 2.0 * q.dimensionless

    def _set_magnification(self, value):
        pass


class Detector(base.Detector):

    modes = Bunch(dict(FAST='fast', HIRES='hires'))

    scintillator = Selection(list(range(1, 12)),
                             help="Current scintillator")

    camera_slot = Selection([1, 2, 3],
                            help="Current camera in use")

    mode = Selection([modes.FAST, modes.HIRES],
                     help="Selected light path mode")

    def __init__(self, cameras):
        if len(cameras) < 3:
            raise ValueError("You must pass at least three cameras")

        super(Detector, self).__init__()

        self._cameras = cameras
        self._highspeed_path = HighSpeedLightpath()
        self._highres_path = HighResolutionLightpath()

        self._set_camera_slot(1)
        self._current_scintillator = 1
        self._current_lightpath = self._highspeed_path

    def _set_camera_slot(self, value):
        self._current_camera = value
        self._camera = self._cameras[value - 1]

    def _get_camera_slot(self):
        return self._current_camera

    def _set_scintillator(self, value):
        self._current_scintillator = value

    def _get_scintillator(self):
        return self._current_scintillator

    def _set_mode(self, value):
        if value == self.modes.FAST:
            self._current_lightpath = self._highspeed_path
        elif value == self.modes.HIRES:
            self._current_lightpath = self._highres_path

    def _get_mode(self):
        if self._current_lightpath == self._highspeed_path:
            return self.modes.FAST

        return self.modes.HIRES

    @property
    def camera(self):
        return self._camera

    @property
    def lightpath(self):
        return self._current_lightpath

    @property
    def magnification(self):
        return self.lightpath.magnification

    @magnification.setter
    def magnification(self, value):
        self.lightpath.magnification = value
