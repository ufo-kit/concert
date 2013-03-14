import quantities as q
from gi.repository import GObject, Uca
from concert.devices.cameras.base import Camera


def _new_setter_wrapper(camera, name):
    def wrapper(value):
        d = {name: value}
        camera.set_properties(**d)

    return wrapper


def _new_getter_wrapper(camera, name):
    def wrapper():
        return camera.get_property(name)

    return wrapper


class UcaCamera(Camera):
    """UcaCamera uses libuca to implement cameras.

    All properties that are exported by a libuca camera is visible in the
    UcaCamera.
    """

    def __init__(self, name):
        super(UcaCamera, self).__init__()

        self._manager = Uca.PluginManager()

        try:
            self._camera = self._manager.get_camerav(name, [])
        except:
            raise ValueError("`{0}' is not a valid camera".format(name))

        units = {
            Uca.Unit.METER: q.m,
            Uca.Unit.SECOND: q.s
        }

        for prop in self._camera.props:
            getter, setter, unit = None, None, None

            if prop.flags & GObject.ParamFlags.READABLE:
                getter = _new_getter_wrapper(self._camera, prop.name)

            if prop.flags & GObject.ParamFlags.WRITABLE:
                setter = _new_setter_wrapper(self._camera, prop.name)

            uca_unit = self._camera.get_unit(prop.name)

            if uca_unit in units:
                unit = units[uca_unit]

            self._register(prop.name, getter, setter, unit)

    def _record_real(self):
        self._camera.start_recording()

    def _stop_real(self):
        self._camera.stop_recording()
