from __future__ import absolute_import
import threading
import sys
import numpy as np

try:
    from gi.repository import Ufo
    import ufo.numpy
except ImportError as e:
    print(str(e))

try:
    from tofu.reco import setup_padding
except ImportError:
    print >> sys.stderr, "You must install tofu to use Ufo features, see "\
                         "'https://github.com/ufo-kit/tofu.git'"

from concert.quantities import q
from concert.coroutines.base import coroutine, inject
from concert.coroutines.filters import sinograms, flat_correct
from concert.coroutines.sinks import Result
from concert.experiments.imaging import tomo_projections_number, frames


class PluginManager(object):

    """Plugin manager that initializes new tasks."""

    def __init__(self):
        self._wrapped = Ufo.PluginManager()

    def get_task(self, name, **kwargs):
        """
        Create a new task from plugin *name* and initialize with *kwargs*.
        """
        task = self._wrapped.get_task(name)
        task.set_properties(**kwargs)
        return task


class InjectProcess(object):

    """Process to inject NumPy data into a UFO processing graph.

    :class:`InjectProcess` can also be used as a context manager, in which
    case it will call :meth:`~.InjectProcess.start` on entering the manager
    and :meth:`~InjectProcess.wait` on exiting it.

    *graph* must either be a Ufo.TaskGraph or a Ufo.TaskNode object.  If it is
    a graph the input tasks will be connected to the roots, otherwise a new
    graph will be created.
    """

    def __init__(self, graph, get_output=False, output_dims=2):
        self.output_tasks = []
        self.sched = Ufo.Scheduler()
        self._started = False

        if isinstance(graph, Ufo.TaskGraph):
            self.graph = graph
            roots = self.graph.get_roots()
        elif isinstance(graph, Ufo.TaskNode):
            self.graph = Ufo.TaskGraph()
            roots = [graph]
        else:
            msg = 'graph is neither Ufo.TaskGraph nor Ufo.TaskNode'
            raise ValueError(msg)

        # Initialize inputs
        self.input_tasks = {}
        self.ufo_buffers = {}
        for root in roots:
            self.input_tasks[root] = []
            self.ufo_buffers[root] = []
            num_inputs = root.get_num_inputs()
            for i in range(num_inputs):
                self.input_tasks[root].append(Ufo.InputTask())
                self.ufo_buffers[root].append(None)
                self.graph.connect_nodes_full(self.input_tasks[root][i], root, i)

        if get_output:
            for i, leave in enumerate(self.graph.get_leaves()):
                self.output_tasks.append(Ufo.OutputTask())
                self.output_tasks[-1].props.num_dims = output_dims
                self.graph.connect_nodes(leave, self.output_tasks[-1])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wait()
        return True

    @coroutine
    def __call__(self, consumer):
        """Co-routine compatible consumer."""
        if not self._started:
            self.start()

        while True:
            item = yield
            self.insert(item)
            consumer.send(self.result(leave_index=0))

    def start(self, arch=None, gpu=None):
        """
        Run the processing in a new thread.

        Use :meth:`.push` to insert data into the processing chaing and
        :meth:`~InjectProcess.wait` to wait until processing has finished."""
        def run_scheduler(sched):
            sched.run(self.graph)

        if arch and gpu:
            sched = Ufo.FixedScheduler()
            sched.set_gpu_nodes(arch, [gpu])
        else:
            sched = self.sched

        self.thread = threading.Thread(target=run_scheduler, args=(sched,))
        self.thread.start()

        if not self._started:
            self._started = True

    def insert(self, array, node=None, index=0):
        """
        Insert *array* into the *node*'s *index* input.

        .. note:: *array* must be a NumPy compatible array.
        """
        if not node:
            if len(self.input_tasks) > 1:
                raise ValueError('input_node cannot be None for graphs with more inputs')
            else:
                node = self.input_tasks.keys()[0]
        if self.ufo_buffers[node][index] is None:
            # reverse shape from rows, cols to x, y
            self.ufo_buffers[node][index] = Ufo.Buffer.new_with_size(array.shape[::-1], None)
        else:
            self.ufo_buffers[node][index] = self.input_tasks[node][index].get_input_buffer()

        self.ufo_buffers[node][index].copy_host_array(array.__array_interface__['data'][0])
        self.input_tasks[node][index].release_input_buffer(self.ufo_buffers[node][index])

    def result(self, leave_index=None):
        """Get result from *leave_index* if not None, all leaves if None. Returns a list of results
        in case *leave_index* is None or one result for the specified leave_index.
        """
        if self.output_tasks:
            indices = range(len(self.output_tasks)) if leave_index is None else [leave_index]
            results = []
            for index in indices:
                buf = self.output_tasks[index].get_output_buffer()
                results.append(np.copy(ufo.numpy.asarray(buf)))
                self.output_tasks[index].release_output_buffer(buf)

            if leave_index is not None:
                results = results[0]

            return results

    def stop(self):
        """Stop input tasks."""
        for input_tasks in self.input_tasks.values():
            for input_task in input_tasks:
                input_task.stop()

    def wait(self):
        """Wait until processing has finished."""
        self.stop()
        self.thread.join()
        self._started = False


class Backproject(InjectProcess):

    """
    Coroutine to reconstruct slices from sinograms using filtered
    backprojection.

    *axis_pos* specifies the center of rotation in pixels within the sinogram.
    If not specified, the center of the image is assumed to be the center of
    rotation.
    """

    def __init__(self, axis_pos=None):
        self.pm = PluginManager()
        self.pad = self.pm.get_task('pad')
        self.crop = self.pm.get_task('crop')
        self.fft = self.pm.get_task('fft', dimensions=1)
        self.ifft = self.pm.get_task('ifft', dimensions=1)
        self.fltr = self.pm.get_task('filter')
        self.backprojector = self.pm.get_task('backproject')

        if axis_pos:
            self.backprojector.props.axis_pos = axis_pos

        super(Backproject, self).__init__(self._connect_nodes(), get_output=True, output_dims=2)

    def _connect_nodes(self, first=None):
        """Connect processing nodes. *first* is the node before fft."""
        graph = Ufo.TaskGraph()
        if first:
            graph.connect_nodes(first, self.pad)

        graph.connect_nodes(self.pad, self.fft)
        graph.connect_nodes(self.fft, self.fltr)
        graph.connect_nodes(self.fltr, self.ifft)
        graph.connect_nodes(self.ifft, self.crop)
        graph.connect_nodes(self.crop, self.backprojector)

        return graph

    @property
    def axis_position(self):
        return self.backprojector.props.axis_pos

    @axis_position.setter
    def axis_position(self, position):
        self.backprojector.set_properties(axis_pos=position)

    def _process(self, sinogram, consumer):
        """Process *sinogram* and send the result to *consumer*. Only to be used in __call__."""
        self.insert(sinogram)
        consumer.send(self.result(leave_index=0))

    @coroutine
    def __call__(self, consumer, arch=None, gpu=None):
        """Get a sinogram, do filtered backprojection and send it to *consumer*."""
        sinogram = yield
        setup_padding(self.pad, self.crop, sinogram.shape[1], sinogram.shape[0])

        if not self._started:
            self.start(arch=arch, gpu=gpu)

        self._process(sinogram, consumer)

        while True:
            sinogram = yield
            self._process(sinogram, consumer)


class FlatCorrectedBackproject(Backproject):

    """
    Coroutine to reconstruct slices from sinograms using filtered
    backprojection. The data are first flat-field corrected and then
    backprojected. All the inputs must be of type unsigned int 16.

    *flat_row* is a row of a flat field, *dark_row* is a row of the dark field.
    The rows must correspond to the sinogram which is being backprojected.
    *axis_pos* specifies the center of rotation in pixels within the sinogram.
    If not specified, the center of the image is assumed to be the center of
    rotation.
    """

    def __init__(self, axis_pos=None, flat_row=None, dark_row=None):
        self.pm = PluginManager()
        self.sino_correction = self.pm.get_task('flat-field-correct')
        self.sino_correction.props.sinogram_input = True

        super(FlatCorrectedBackproject, self).__init__(axis_pos=axis_pos)

        self.flat_row = flat_row
        self.dark_row = dark_row

    def _connect_nodes(self):
        """Connect nodes with flat-correction."""
        return super(FlatCorrectedBackproject, self)._connect_nodes(first=self.sino_correction)

    @property
    def axis_position(self):
        return self.backprojector.props.axis_pos

    @axis_position.setter
    def axis_position(self, position):
        self.backprojector.props.axis_pos = position

    @property
    def dark_row(self):
        return self._dark_row

    @dark_row.setter
    def dark_row(self, row):
        if row is not None:
            row = row.astype(np.float32)

        self._dark_row = row

    @property
    def flat_row(self):
        return self._flat_row

    @flat_row.setter
    def flat_row(self, row):
        if row is not None:
            row = row.astype(np.float32)

        self._flat_row = row

    def _process(self, sinogram, consumer):
        self.insert(sinogram.astype(np.float32), node=self.sino_correction, index=0)
        if self.dark_row is None or self.flat_row is None:
            raise ValueError('Both flat and dark rows must be set')
        self.insert(self.dark_row, node=self.sino_correction, index=1)
        self.insert(self.flat_row, node=self.sino_correction, index=2)
        consumer.send(self.result(leave_index=0))


@coroutine
def middle_row(consumer):
    while True:
        frame = yield
        row = frame.shape[0] / 2
        part = frame[row-1:row+1, :]
        consumer.send(part)


def center_rotation_axis(camera, motor, initial_motor_step,
                         num_iterations=2, num_projections=None, flat=None, dark=None):
    """
    Center the rotation axis controlled by *motor*.

    Use an iterative approach to center the rotation axis. Around *motor*s
    current position, we evaluate five points by running a reconstruction.
    *rotation_motor* rotates the sample around the tomographic axis.
    *num_iterations* controls the final resolution of the step size, halving
    each iteration. *flat* is a flat field frame and *dark* is a dark field
    frame which will be used for flat correcting the acuired projections.
    """

    width_2 = camera.roi_width.magnitude / 2.0
    axis_pos = width_2

    # Crop the dark and flat
    if flat is not None:
        middle = flat.shape[0] / 2
        flat = flat[middle, :]
        if dark is not None:
            dark = dark[middle, :]

    n = num_projections or tomo_projections_number(camera.roi_width)
    angle_step = np.pi / n * q.rad

    step = initial_motor_step
    current = motor.position

    for i in range(num_iterations):
        frm = current - step
        to = current + step
        div = 2.0 * step / 5.0

        positions = (frm, frm + div, current, current + div, to)
        scores = []

        for position in positions:
            motor.position = position
            backproject = Backproject(axis_pos)
            sino_result = Result()
            sino_coro = sino_result()
            if flat is not None:
                sino_coro = flat_correct(flat, sino_coro, dark=dark)

            inject(frames(n, camera, callback=lambda: rotation_motor.move(angle_step).join()),
                   middle_row(sinograms(n, sino_coro)))

            sinogram = (sinogram.result[0, :, :], )
            result = Result()
            m0 = np.mean(np.sum(sinogram[0], axis=1))

            inject(sinogram, backproject(result()))
            backproject.wait()

            img = result.result

            # Other possibilities: sum(abs(img)) or sum(img * heaviside(-img))
            score = np.sum(np.abs(np.gradient(img))) / m0
            scores.append(score)

        current = positions[scores.index(min(scores))]
        step /= 2.0


def compute_rotation_axis(sinogram, initial_step=None, max_iterations=14,
                          slice_consumer=None, score_consumer=None):

    width_2 = sinogram.shape[1] / 2.0
    iteration = 0
    step = initial_step or width_2 / 2
    current = width_2

    while step > 1 and iteration < max_iterations:
        frm = current - step
        to = current + step
        div = 2.0 * step / 5.0

        axes = (frm, frm + div, current, current + div, to)
        scores = []

        for axis in axes:
            backproject = Backproject(axis)
            result = Result()

            inject((sinogram, ), backproject(result()))
            backproject.wait()

            img = result.result

            # Other possibilities: sum(abs(img)) or sum(img * heaviside(-img))
            score = np.sum(np.abs(np.gradient(img)))
            scores.append(score)
            if slice_consumer:
                slice_consumer.send(img)
            if score_consumer:
                score_consumer.send(axis * q.px)

        current = axes[scores.index(min(scores))]
        step /= 2.0
        iteration += 1

    return current
