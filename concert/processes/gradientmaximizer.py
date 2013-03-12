import logging

from gi.repository import Ufo, Uca


class GradientMaximizer(object):
    def __init__(self, camera, measure, axis, limits, config=None):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        self.axis = axis 
        self.min_limit, self.max_limit = limits

        self.step = 1.0
        self.stopped = False
        self.measure = measure

        self.camera = camera
        self.camera.props.trigger_mode = Uca.CameraTrigger.INTERNAL

        # It's a bit unfortunate, that we have to keep a reference to the
        # plugin manager ourselves. But if we don't it will be freed by
        # the Python run-time and thus all plugins will crash.
        self.pm = Ufo.PluginManager(config=config)
        self.graph = Ufo.TaskGraph()

        self.camera_task = self.pm.get_task('camera')
        self.camera_task.set_properties(camera=camera, count=5)

        width, height = camera.props.roi_width, camera.props.roi_height
        window = 100 / 2
        roi_x = width / 2 - window
        roi_y = height / 2 - window

        self.roi = self.pm.get_task('region-of-interest')
        self.roi.set_properties(x=roi_x, y=roi_y, width=window, height=window)

        self.measure_task = self.pm.get_task('sharpness-measure')
        self.measure_task.connect('notify::sharpness', self._evaluate)

        self.graph.connect_nodes(self.camera_task, self.roi)
        self.graph.connect_nodes(self.roi, self.measure_task)
        self.sched = Ufo.Scheduler(config=config)

    def start(self):
        self.logger.info('Move motor to position 0')
        self.axis.set_position(self.min_limit)

        self.logger.info('Start camera recording')
        self.camera.start_recording()
        self.camera.trigger()

        self.logger.info('Start image processing pipeline')
        self.sched.run(self.graph)

    def stop(self):
        self.logger.info('Stop measurement')

    def _gradient_step(self, value):
        if self.measure.is_better(value) or self.step == 1.0:
            return self.step / 2.0

        return self.step - self.step / 2.0

    def _get_absolute_position(self, x):
        return (self.max_limit - self.min_limit) * x - self.min_limit

    def _evaluate(self, obj, param):
        print('Checking {0}'.format(obj.props.sharpness))

        if self.stopped:
            return

        if self.measure.set_point_reached(obj.props.sharpness):
            self.camera.stop_recording()
            self.stop()
            self.stopped = True
            return

        self.measure.value = obj.props.sharpness
        self.step = self._gradient_step(obj.props.sharpness)

        position = self._get_absolute_position(self.step)
        self.axis.set_position(position, blocking=True)
        self.camera.trigger()
