"""
Cameras supported by the libuca library.
"""
import functools
import logging
import time
import numpy as np
from concert.quantities import q
from concert.base import Parameter, Quantity
from concert.helpers import Bunch
from concert.devices.cameras import base


LOG = logging.getLogger(__name__)


def _new_setter_wrapper(name, unit=None):
    def _wrapper(instance, value):
        if unit:
            value = value.to(unit)

        try:
            dic = {name: value.magnitude}
        except AttributeError:
            dic = {name: value}

        instance.uca.set_properties(**dic)

    return _wrapper


def _new_getter_wrapper(name, unit=None):
    def _wrapper(instance):
        value = instance.uca.get_property(name)

        if unit:
            return value * unit

        return value

    return _wrapper


def _translate_gerror(func):
    from gi.repository import GLib

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GLib.GError as ge:
            raise base.CameraError(str(ge))

    return _wrapper


class Camera(base.Camera):

    """libuca-based camera.

    All properties that are exported by the underlying camera are also visible.
    """

    def __init__(self, name):
        """
        Create a new libuca camera.

        The *name* is passed to the uca plugin manager.

        :raises ValueError: In case camera *name* does not exist.
        """

        super(Camera, self).__init__()

        from gi.repository import GObject, Uca

        self._manager = Uca.PluginManager()

        try:
            self.uca = self._manager.get_camerav(name, [])
        except:
            raise ValueError("`{0}' is not a valid camera".format(name))

        units = {
            Uca.Unit.METER: q.m,
            Uca.Unit.SECOND: q.s,
            Uca.Unit.DEGREE_CELSIUS: q.celsius,
            Uca.Unit.COUNT: q.dimensionless,
            Uca.Unit.PIXEL: q.pixel,
        }

        parameters = {}

        for prop in self.uca.props:
            if prop.name == 'trigger-mode' or prop.name == 'frames-per-second':
                continue

            getter, setter, unit = None, None, None

            uca_unit = self.uca.get_unit(prop.name)

            if uca_unit in units:
                unit = units[uca_unit]

            if prop.flags & GObject.ParamFlags.READABLE:
                getter = _new_getter_wrapper(prop.name, unit)

            if prop.flags & GObject.ParamFlags.WRITABLE:
                setter = _new_setter_wrapper(prop.name, unit)

            name = prop.name.replace('-', '_')

            if uca_unit in units:
                parameters[name] = Quantity(unit, fget=getter, fset=setter, help=prop.blurb)
            else:
                parameters[name] = Parameter(fget=getter, fset=setter, help=prop.blurb)

        if parameters:
            self.install_parameters(parameters)

        class _Dummy(object):
            pass

        setattr(self.uca, 'enum_values', _Dummy())

        def get_enum_bunch(enum):
            enum_map = {}

            for key, value in enum.__enum_values__.items():
                name = value.value_nick.upper().replace('-', '_')
                enum_map[name] = key

            return Bunch(enum_map)

        for prop in self.uca.props:
            if hasattr(prop, 'enum_class'):
                setattr(self.uca.enum_values, prop.name.replace('-', '_'),
                        get_enum_bunch(prop.default_value))

        self._uca_get_frame_rate = _new_getter_wrapper('frames-per-second')
        self._uca_set_frame_rate = _new_setter_wrapper('frames-per-second')

        # Invert the uca trigger mode dict in order to return concert values
        trigger_dict = self.uca.enum_values.trigger_mode.__dict__
        self._uca_to_concert_trigger = {v: k for k, v in trigger_dict.items()}
        self._uca_get_trigger = _new_getter_wrapper('trigger-mode')
        self._uca_set_trigger = _new_setter_wrapper('trigger-mode')

        self._record_shape = None
        self._record_dtype = None

    def _get_frame_rate(self):
        return self._uca_get_frame_rate(self) / q.s

    def _set_frame_rate(self, frame_rate):
        self._uca_set_frame_rate(self, frame_rate * q.s)

    def _get_trigger_mode(self):
        uca_trigger = self._uca_get_trigger(self)
        return self._uca_to_concert_trigger[uca_trigger]

    def _set_trigger_mode(self, mode):
        uca_value = getattr(self.uca.enum_values.trigger_mode, mode)
        self._uca_set_trigger(self, uca_value)

    @_translate_gerror
    def _record_real(self):
        self._record_shape = self.roi_height.magnitude, self.roi_width.magnitude
        self._record_dtype = np.uint16 if self.sensor_bitdepth.magnitude > 8 else np.uint8
        self.uca.start_recording()

    @_translate_gerror
    def _stop_real(self):
        self.uca.stop_recording()

    @_translate_gerror
    def _trigger_real(self):
        self.uca.trigger()

    @_translate_gerror
    def _grab_real(self):
        array = np.empty(self._record_shape, dtype=self._record_dtype)
        data = array.__array_interface__['data'][0]

        if self.uca.grab(data):
            return array

        return None


class Pco(Camera):

    """Pco camera implemented by libuca."""

    def __init__(self):
        super(Pco, self).__init__('pco')

    def stream(self, consumer):
        """stream frames to the *consumer*."""
        try:
            self.acquire_mode = self.uca.enum_values.acquire_mode.AUTO
            self.storage_mode = self.uca.enum_values.storage_mode.RECORDER
            self.record_mode = self.uca.enum_values.record_mode.RING_BUFFER
        except:
            pass

        return super(Pco, self).stream(consumer)


class PCO4000(Pco):

    """PCO.4000 camera implementation."""

    _ALLOWED_DURING_RECORDING = ['trigger', '_trigger_real', '_last_grab_time',
                                 'grab', '_grab_real', '_record_shape', '_record_dtype',
                                 'uca', 'stop_recording']

    def __init__(self):
        self._lock_access = False
        self._last_grab_time = None
        super(PCO4000, self).__init__()

    def start_recording(self):
        super(PCO4000, self).start_recording()
        self._lock_access = True

    def stop_recording(self):
        self._lock_access = False
        super(PCO4000, self).stop_recording()

    def _grab_real(self):
        # For the PCO.4000 there must be a delay of at least one second
        # between consecutive grabs, otherwise it crashes. We provide
        # appropriate timeout here.
        current = time.time()
        if self._last_grab_time and current - self._last_grab_time < 1:
            time.sleep(1 - current + self._last_grab_time)
        result = super(PCO4000, self)._grab_real()
        self._last_grab_time = time.time()

        return result

    def __getattribute__(self, name):
        if object.__getattribute__(self, '_lock_access') and name \
           not in PCO4000._ALLOWED_DURING_RECORDING:
            raise AttributeError('PCO.4000 is inaccessible during recording')
        return object.__getattribute__(self, name)


class Dimax(Pco, base.BufferedMixin):

    """A pco.dimax camera implementation."""

    def __init__(self):
        super(Dimax, self).__init__()

    def _readout_real(self, num_frames=None):
        """Readout *num_frames* frames."""
        if num_frames is None:
            num_frames = self.recorded_frames

        if not 0 < num_frames <= self.recorded_frames:
            raise ValueError("Number of frames {} ".format(num_frames) +
                             "must be more than zero and less than the recorded " +
                             "number of frames {}".format(self.recorded_frames))

        try:
            self.uca.start_readout()

            for i in xrange(num_frames):
                yield self.grab()
        finally:
            self.uca.stop_readout()
