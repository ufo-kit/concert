"""
Cameras supported by the libuca library.
"""
import logging
import time
import numpy as np
from concert.coroutines import null
from concert.helpers import async, inject
from concert.quantities import q
from concert.base import Parameter
from concert.helpers import Bunch
from concert.devices.cameras import base


LOG = logging.getLogger(__name__)


def _new_setter_wrapper(camera, name, unit=None):
    def _wrapper(value):
        if unit:
            value = value.to(unit)

        try:
            dic = {name: value.magnitude}
        except AttributeError:
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
    array = np.empty(dims, dtype=dtype)
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

        try:
            self.uca = self._manager.get_camerav(name, [])
        except:
            raise ValueError("`{0}' is not a valid camera".format(name))

        units = {
            Uca.Unit.METER: q.m,
            Uca.Unit.SECOND: q.s,
            Uca.Unit.DEGREE_CELSIUS: q.celsius,
            Uca.Unit.COUNT: q.count,
            Uca.Unit.PIXEL: q.count,
        }

        parameters = []

        for prop in self.uca.props:
            getter, setter, unit = None, None, None

            uca_unit = self.uca.get_unit(prop.name)

            if uca_unit in units:
                unit = units[uca_unit]

            if prop.flags & GObject.ParamFlags.READABLE:
                getter = _new_getter_wrapper(self.uca, prop.name, unit)

            if prop.flags & GObject.ParamFlags.WRITABLE:
                setter = _new_setter_wrapper(self.uca, prop.name, unit)

            parameter = Parameter(prop.name, getter, setter, unit)
            parameters.append(parameter)

        super(Camera, self).__init__(parameters)

    def readout(self, condition=lambda: True):
        """
        Readout images from the camera buffer. *condition* is a callable,
        as long as it resolves to True the camera keeps grabbing.
        """
        while condition():
            image = None
            try:
                image = self.grab()
            except Exception as exc:
                LOG.debug("An error {} occured".format(exc) +
                          " during readout, stopping")
            if image is None:
                break
            yield image

    def acquire(self, num_frames):
        """
        Acquire *num_frames* frames. The camera is triggered explicitly from
        Concert so the number of recorded frames is exact. The frames are
        yielded as they are being acquired.
        """
        try:
            self.trigger_mode = self.uca.enum_values.trigger_mode.SOFTWARE
            self.start_recording()
            for i in xrange(num_frames):
                self.trigger()
                yield self.grab()
        finally:
            self.stop_recording()

    def _get_frame_rate(self):
        return self.frames_per_second / q.s

    def _set_frame_rate(self, frame_rate):
        self.frames_per_second = frame_rate * q.s

    def _record_real(self):
        self.uca.start_recording()

    def _stop_real(self):
        self.uca.stop_recording()

    def _trigger_real(self):
        self.uca.trigger()

    def _grab_real(self):
        array, data = _create_data_array(self.uca)

        if self.uca.grab(data):
            return array

        return None


class Pco(Camera):

    def __init__(self):
        super(Pco, self).__init__('pco')

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

    @async
    def freerun(self, consumer):
        """Start recording and send live frames to *consumer*."""
        self.trigger_mode = self.uca.enum_values.trigger_mode.AUTO
        try:
            self.storage_mode = self.uca.enum_values.storage_mode.RECORDER
            self.record_mode = self.uca.enum_values.record_mode.RING_BUFFER
        except:
            pass
        self.start_recording()
        inject(self.readout(lambda: self.uca.props.is_recording), consumer)


class Dimax(Pco):

    """A PCO.dimax camera implementation based on libuca :py:class:`Camera`."""

    def __init__(self):
        super(Dimax, self).__init__()

    def readout_blocking(self, condition=lambda: True):
        """
        Readout the frames and don't allow recording in the meantime.
        *condition* is the same as in :py:meth:`Camera.readout`.
        """
        try:
            self.uca.start_readout()
            for frame in super(Dimax, self).readout(condition):
                yield frame
        finally:
            self.uca.stop_readout()

    def record_auto(self, num_frames):
        """
        Record approximately *num_frames* frames into the internal camera
        buffer. Camera is set to a mode when it is triggered automatically and
        live images streaming is enabled.  Live frames are yielded as they are
        being grabbed.

        **Note** the number of actually recorded images may differ.
        """
        @async
        def async_wait():
            try:
                sleep_time = (num_frames /
                              self.frame_rate).to_base_units().magnitude
                time.sleep(sleep_time)
            finally:
                self.stop_recording()

        self.trigger_mode = self.uca.enum_values.trigger_mode.AUTO
        self.storage_mode = self.uca.enum_values.storage_mode.RECORDER
        self.record_mode = self.uca.enum_values.record_mode.SEQUENCE

        # We need to make sure that the camera is in recording mode when we
        # start grabbing live frames, thus we start it synchronously.
        self.start_recording()

        # The sleeping and camera stopping can be handled asynchronously
        acq_future = async_wait()

        # Yield live frames
        for frame in self.readout(condition=lambda:
                                  self.uca.props.is_recording):
            yield frame
            time.sleep(0.001)

        # Wait for the acquisition to end
        acq_future.result()

    def acquire_auto(self, num_frames, consumer=None):
        """
        Acquire and readout *num_frames*. Frames are first recorded to the
        internal camera memory and then read out. The camera is triggered
        automatically. *consumer* is a coroutine which is fed with live
        frames. After the recording is done the frames are yielded as they are
        being grabbed from the camera.
        """
        # We need to provide a consumer, otherwise the generator method
        # wouldn't start
        consumer = null() if consumer is None else consumer
        inject(self.record_auto(num_frames), consumer)

        return (frame for frame in self.readout_blocking())
