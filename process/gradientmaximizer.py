import logging

from gi.repository import Ufo, Uca

class GradientMaximizer(object):
    def __init__(self, camera, measure, controller, config=None):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        self._controller = controller
        self._min_limit, self._max_limit = controller.get_limits()

        self._step = 1.0
        self._stopped = False
        self._measure = measure

        self._camera = camera
        self._camera.props.trigger_mode = Uca.CameraTrigger.INTERNAL
        
        # It's a bit unfortunate, that we have to keep a reference to the plugin
        # manager ourselves. But if we don't it will be freed by the Python
        # run-time and thus all plugins will crash.
        self._pm = Ufo.PluginManager(config=config)
        self._graph = Ufo.TaskGraph()

        self._camera_task = self._pm.get_task('camera')
        self._camera_task.set_properties(camera=camera, count=5)

        width, height = camera.props.roi_width, camera.props.roi_height
        window = 100 / 2

        self._roi = self._pm.get_task('region-of-interest')
        self._roi.set_properties(x=width/2-window, y=height/2-window,
                                 width=window, height=window)

        self._measure_task = self._pm.get_task('sharpness-measure')
        self._measure_task.connect('notify::sharpness', self._evaluate)

        self._graph.connect_nodes(self._camera_task, self._roi)
        self._graph.connect_nodes(self._roi, self._measure_task)
        self._sched = Ufo.Scheduler(config=config)

    def start(self):
        self.logger.info('Move motor to position 0')
        self._controller.move_to_relative_position(0.0)

        self.logger.info('Start camera recording')
        self._camera.start_recording()
        self._camera.trigger()

        self.logger.info('Start image processing pipeline')
        self._sched.run(self._graph)
        
    def stop(self):
        self.logger.info('Stop measurement')

    def _gradient_step(self, value):
        if self._measure.is_better(value) or self._step == 1.0:
            return self._step / 2.0

        return self._step - self._step / 2.0

    def _evaluate(self, obj, param):
        print('Checking {0}'.format(obj.props.sharpness))

        if self._stopped:
            return

        if self._measure.set_point_reached(obj.props.sharpness):
            self._camera.stop_recording()
            self.stop()
            self._stopped = True
            return

        self._measure.last_value = obj.props.sharpness
        self._step = self._gradient_step(obj.props.sharpness)
        self._controller.move_to_relative_position(self._step)
        self._camera.trigger()

