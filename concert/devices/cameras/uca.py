"""
Cameras supported by the libuca library.
"""
import numpy as np
from concert.quantities import q
from concert.base import Parameter
from concert.devices.cameras import base


def _new_setter_wrapper(camera, name, unit=None):
    def _wrapper(value):
        if unit:
            value = value.to(unit)

        dic = {name: value}
        camera.set_properties(**dic)

    return _wrapper


def _new_getter_wrapper(camera, name, unit=None):
    def _wrapper():
        value = camera.get_property(name)

        if unit:
            return value * unit

        return value

    return _wrapper


def _create_data_array(camera):
    bits = camera.props.sensor_bitdepth
    dtype = np.uint16 if bits > 8 else np.uint8
    dims = camera.props.roi_height, camera.props.roi_width
    array = np.zeros(dims, dtype=dtype)
    return (array, array.__array_interface__['data'][0])


class Camera(base.Camera):

    """libuca-based camera.

    All properties that are exported by the underlying camera are also visible
    in :class:`UcaCamera`.

    :raises ValueError: In case camera *name* does not exist.
    """

    def __init__(self, name):
        from gi.repository import GObject, Uca

        self._manager = Uca.PluginManager()
        self._data = None
        self._array = None

        try:
            self.camera = self._manager.get_camerav(name, [])
        except:
            raise ValueError("`{0}' is not a valid camera".format(name))

        units = {
            Uca.Unit.METER: q.m,
            Uca.Unit.SECOND: q.s,
            Uca.Unit.DEGREE_CELSIUS: q.celsius,
            Uca.Unit.COUNT: q.count
        }

        parameters = []

        for prop in self.camera.props:
            getter, setter, unit = None, None, None

            uca_unit = self.camera.get_unit(prop.name)

            if uca_unit in units:
                unit = units[uca_unit]

            if prop.flags & GObject.ParamFlags.READABLE:
                getter = _new_getter_wrapper(self.camera, prop.name, unit)

            if prop.flags & GObject.ParamFlags.WRITABLE:
                setter = _new_setter_wrapper(self.camera, prop.name, unit)

            parameter = Parameter(prop.name, getter, setter, unit)
            parameters.append(parameter)

        super(Camera, self).__init__(parameters)

    def _record_real(self):
        self._array, self._data = _create_data_array(self.camera)
        self.camera.start_recording()

    def _stop_real(self):
        self.camera.stop_recording()

    def _trigger_real(self):
        self.camera.trigger()

    def _grab_real(self):
        if not self._data:
            self._array, self._data = _create_data_array(self.camera)

        self.camera.grab(self._data)
        return self._array
