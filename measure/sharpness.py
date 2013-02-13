import logging

from gi.repository import Ufo


class Sharpness(object):
    def __init__(self, camera, config=None):
        self._callbacks = []

        self.logger = logging.getLogger('process.measure.sharpness')
        self.logger.propagate = True

        # It's a bit unfortunate, that we have to keep a reference to the plugin
        # manager ourselves. But if we don't it will be freed by the Python
        # run-time and thus all plugins will crash.
        self._pm = Ufo.PluginManager(config=config)
        self._graph = Ufo.TaskGraph()

        self._camera = self._pm.get_task('camera')
        self._camera.set_properties(camera=camera, count=5)

        width, height = camera.props.roi_width, camera.props.roi_height
        window = 100 / 2

        self._roi = self._pm.get_task('region-of-interest')
        self._roi.set_properties(x=width/2-window, y=height/2-window,
                                 width=window, height=window)

        self._measure = self._pm.get_task('sharpness-measure')
        self._measure.connect('notify::sharpness', self._on_sharpness_changed)

        self._graph.connect_nodes(self._camera, self._roi)
        self._graph.connect_nodes(self._roi, self._measure)
        self._sched = Ufo.Scheduler(config=config)

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def start(self):
        self._sched.run(self._graph)

    def stop(self):
        self.logger.info('Stop measurement')
        pass

    def _on_sharpness_changed(self, obj, param):
        for callback in self._callbacks:
            callback(obj.props.sharpness)
