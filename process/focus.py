import logging


class Focus(object):
    def __init__(self, camera, measure, controller):

        self.logger = logging.getLogger('process.focus')
        self.logger.propagate = True

        self._last_value = -1000000.0
        self._controller = controller
        self._camera = camera
        self._step = 1.0
        self._stopped = False
        self._measure = measure
        self._measure.register_callback(self._evaluate)

    def start(self):
        self.logger.info('Move motor to position 0')
        self._controller.move_to_relative_position(0.0)

        self.logger.info('Start camera recording')
        self._camera.start_recording()

        self.logger.info('Start measurement procedure')
        self._measure.start()

    def _is_better(self, value):
        # This concrete check should actually be moved to some other evaluator place
        return self._last_value < value

    def _set_point_reached(self, value):
        # This concrete check should actually be moved to some other evaluator place
        return abs(self._last_value - value) < 0.01

    def _gradient_step(self, value):
        if self._is_better(value) or self._step == 1.0:
            return self._step / 2.0

        return -self._step / 2.0

    def _evaluate(self, value):
        print('Checking {0}'.format(value))

        if self._stopped:
            return

        if self._set_point_reached(value):
            self._camera.stop_recording()
            self._measure.stop()
            self._stopped = True
            return

        self._last_value = value
        self._step = self._gradient_step(value)
        self._controller.move_to_relative_position(self._step)

