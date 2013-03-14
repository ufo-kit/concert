import quantities as q
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
    """Provide a libuca-based camera called *name*.

    All properties that are exported by the underlying camera are also visible
    in :class:`UcaCamera`.

    :raises ValueError: In case camera *name* does not exist.
    """

    def __init__(self, name):
        super(UcaCamera, self).__init__()

        from gi.repository import GObject, Uca

        self._manager = Uca.PluginManager()

        try:
            self.camera = self._manager.get_camerav(name, [])
        except:
            raise ValueError("`{0}' is not a valid camera".format(name))

        units = {
            Uca.Unit.METER: q.m,
            Uca.Unit.SECOND: q.s
        }

        for prop in self.camera.props:
            getter, setter, unit = None, None, None

            if prop.flags & GObject.ParamFlags.READABLE:
                getter = _new_getter_wrapper(self.camera, prop.name)

            if prop.flags & GObject.ParamFlags.WRITABLE:
                setter = _new_setter_wrapper(self.camera, prop.name)

            uca_unit = self.camera.get_unit(prop.name)

            if uca_unit in units:
                unit = units[uca_unit]

            self._register(prop.name, getter, setter, unit)

    def _record_real(self):
        self.camera.start_recording()

    def _stop_real(self):
        self.camera.stop_recording()
