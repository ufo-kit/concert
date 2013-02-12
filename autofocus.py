from gi.repository import Ufo, Uca

class FocusMeasureProcess(object):
    def __init__(self, camera):
        self._callbacks = []

        # It's a bit unfortunate, that we have to keep a reference to the plugin
        # manager ourselves. But if we don't it will be freed by the Python
        # run-time and thus all plugins will crash.
        self._pm = Ufo.PluginManager()
        self._graph = Ufo.TaskGraph()

        self._uca_pm = Uca.PluginManager()
        
        self._camera = self._pm.get_task('camera')
        self._camera.set_properties(camera=camera)

        self._roi = self._pm.get_task('region-of-interest')
        self._roi.set_properties(x=100, y=100, width=100, height=100)

        self._measure = self._pm.get_task('sharpness-measure')
        self._measure.connect('notify::sharpness', self._on_sharpness_changed)

        self._graph.connect_nodes(self._camera, self._roi)
        self._graph.connect_nodes(self._roi, self._measure)
        self._sched = Ufo.Scheduler()

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def start(self):
        self._sched.run(self._graph)

    def stop(self):
        pass

    def _on_sharpness_changed(self, obj, param):
        for callback in self._callbacks:
            callback(obj.props.sharpness)


class FocusControlProcess(object):
    def __init__(self, camera, controller=None):
        self._last_value = -1000000.0
        self._controller = controller
        self._camera = camera
        self._measure = FocusMeasureProcess(camera)
        self._measure.register_callback(self._evaluate)

    def start(self):
        self._camera.start_recording()
        self._measure.start()

    def _set_point_reached(value):
        return abs(self._last_value - value) < 0.01

    def _evaluate(self, value):
        if self._last_value < value:
            if self._set_point_reached(value):
                self._camera.stop_recording()
                self._measure.stop()
            else:
                self._last_value = value
                print('>> Moving controller to new position according to {0}'.format(value))



if __name__ == '__main__':
    pm = Uca.PluginManager()
    camera = pm.get_camera('mock')

    control = FocusControlProcess(camera)
    control.start()
