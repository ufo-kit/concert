"""
Cameras supported by the libuca library.
"""
import functools
import logging
import numpy as np
from concert.quantities import q
from concert.base import Parameter, Quantity
from concert.helpers import Bunch
from concert.devices.cameras import base


LOG = logging.getLogger(__name__)
q.autoconvert_offset_to_baseunit = True


def _new_setter_wrapper(name, unit=None):
    def _wrapper(instance, value):
        if instance.state == 'recording':
            raise base.CameraError('Changing parameters is not allowed while recording')

        if unit and str(unit) != "delta_degC":
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
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        from gi.repository import GLib
        try:
            return func(*args, **kwargs)
        except GLib.GError as ge:
            raise base.CameraError(str(ge))

    return _wrapper


class Camera(base.Camera):

    """libuca-based camera.

    All properties that are exported by the underlying camera are also visible.
    """

    def __init__(self, name, params=None):
        """
        Create a new libuca camera.

        The *name* is passed to the uca plugin manager.

        :raises CameraError: In case camera *name* does not exist.
        """

        super(Camera, self).__init__()

        from gi.repository import GObject, GLib, Uca

        self._manager = Uca.PluginManager()

        params = params if params else {}

        try:
            self.uca = self._manager.get_camerah(name, params)
        except GLib.GError as ge:
            raise base.CameraError(str(ge))
        except:
            raise base.CameraError("`{0}' is not a valid camera".format(name))

        units = {
            Uca.Unit.METER: q.m,
            Uca.Unit.SECOND: q.s,
            Uca.Unit.DEGREE_CELSIUS: q.delta_degC,
            Uca.Unit.COUNT: q.dimensionless,
            Uca.Unit.PIXEL: q.pixel,
        }

        parameters = {}

        for prop in self.uca.props:
            if prop.name in ('trigger-source', 'trigger-type', 'frames-per-second'):
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

        # Invert the uca trigger source dict in order to return concert values
        trigger_dict = self.uca.enum_values.trigger_source.__dict__
        self._uca_to_concert_trigger = {v: k for k, v in trigger_dict.items()}
        self._uca_get_trigger = _new_getter_wrapper('trigger-source')
        self._uca_set_trigger = _new_setter_wrapper('trigger-source')

        self._record_shape = None
        self._record_dtype = None

    @transition(target='readout')
    def start_readout(self):
        self.uca.start_readout()

    @transition(target='standby')
    def stop_readout(self):
        self.uca.stop_readout()

    def grab(self, index=None):
        return self.convert(self._grab_real(index))

    def write(self, name, data):
        """Write NumPy array *data* for *name*."""
        raw = data.__array_interface__['data'][0]
        self.uca.write(name, raw, data.nbytes)

    def _get_frame_rate(self):
        return self._uca_get_frame_rate(self) / q.s

    def _set_frame_rate(self, frame_rate):
        self._uca_set_frame_rate(self, frame_rate * q.s)

    def _get_trigger_source(self):
        uca_trigger = self._uca_get_trigger(self)
        return self._uca_to_concert_trigger[uca_trigger]

    def _set_trigger_source(self, source):
        uca_value = getattr(self.uca.enum_values.trigger_source, source)
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
    def _grab_real(self, index=None):
        array = np.empty(self._record_shape, dtype=self._record_dtype)
        data = array.__array_interface__['data'][0]

        if index is not None:
            if self.uca.readout(data, index):
                return array
            else:
                raise base.CameraError('No frame available')

        if self.uca.grab(data):
            return array
        else:
            raise base.CameraError('No frame available')
