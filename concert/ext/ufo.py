from __future__ import absolute_import
import copy
import logging
import threading
import time
import sys
import numpy as np

try:
    import gi
    gi.require_version('Ufo', '0.0')
    from gi.repository import Ufo
    import ufo.numpy
except ImportError as e:
    print(str(e))

try:
    from tofu.config import SECTIONS, UNI_RECO_PARAMS
    from tofu.util import setup_padding, get_reconstruction_regions
    from tofu.unireco import (CTGeometry, setup_graph, set_projection_filter_scale, make_runs,
                              DTYPE_CL_SIZE)
    from tofu.tasks import get_task
except ImportError:
    print >> sys.stderr, "You must install tofu to use Ufo features, see "\
                         "'https://github.com/ufo-kit/tofu.git'"

from multiprocessing.pool import ThreadPool
from concert.quantities import q
from concert.coroutines.base import coroutine, inject
from concert.coroutines.filters import sinograms, flat_correct
from concert.coroutines.sinks import Result
from concert.experiments.imaging import tomo_projections_number, frames


LOG = logging.getLogger(__name__)


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
    graph will be created. *scheduler* is one of the ufo schedulers, e.g.
    Ufo.Scheduler or Ufo.FixedScheduler.
    """

    def __init__(self, graph, get_output=False, output_dims=2, scheduler=None, copy_inputs=False):
        self.output_tasks = []
        self.sched = scheduler if scheduler else Ufo.Scheduler()
        self._started = False
        self.copy_inputs = copy_inputs

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

        if array is not None:
            if self.copy_inputs:
                array = np.copy(array, order='C')
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


class UniversalBackprojectArgs(object):
    def __init__(self, width, height, center_position_x, center_position_z, number, overall_angle=np.pi):
        for section in UNI_RECO_PARAMS:
            for arg in SECTIONS[section]:
                settings = SECTIONS[section][arg]
                default = settings['default']
                if default is not None and 'type' in settings:
                    default = settings['type'](default)
                setattr(self, arg.replace('-', '_'), default)
        self.y = 0
        self.width = width
        self.height = height
        self.center_position_x = center_position_x
        self.center_position_z = center_position_z
        self.number = number
        self.overall_angle = overall_angle
        self._slice_metric = None

    @property
    def slice_metric(self):
        return self._slice_metric

    @slice_metric.setter
    def slice_metric(self, metric):
        if metric not in [None, 'min', 'max', 'sum', 'mean', 'var', 'std',
                          'skew', 'kurtosis', 'msag']:
            raise UniversalBackprojectArgsError("Metric '{}' not known".format(metric))
        self._slice_metric = metric

    @property
    def z_parameter(self):
        return self._z_parameter

    @z_parameter.setter
    def z_parameter(self, name):
        if name not in SECTIONS['universal-reconstruction']['z-parameter']['choices']:
            raise UniversalBackprojectArgsError("Unknown z parameter '{}'".format(name))
        self._z_parameter = name


class UniversalBackproject(InjectProcess):
    def __init__(self, args, resources=None, gpu_index=0, flat=None, dark=None, region=None,
                 copy_inputs=False, before_download_event=None):
        self.before_download_event = before_download_event
        scheduler = Ufo.FixedScheduler()
        if resources:
            scheduler.set_resources(resources)
        gpu = scheduler.get_resources().get_gpu_nodes()[gpu_index]

        self.args = copy.deepcopy(args)
        x_region, y_region, z_region = get_reconstruction_regions(self.args, store=True)
        set_projection_filter_scale(self.args)
        if region is not None:
            self.args.region = region
        LOG.debug('Creating reconstructor for gpu %d, region: %s', gpu_index, self.args.region)
        geometry = CTGeometry(self.args)
        geometry.optimize_args()
        self.args = geometry.args
        self.dark = dark
        self.flat = flat
        if self.dark is not None and self.flat is not None:
            LOG.debug('Flat correction on')
            self.dark = self.dark[self.args.y:self.args.y + self.args.height].astype(np.float32)
            self.flat = self.flat[self.args.y:self.args.y + self.args.height].astype(np.float32)

        regions = make_runs([gpu], x_region, y_region, self.args.region,
                            DTYPE_CL_SIZE[self.args.store_type],
                            slices_per_device=self.args.slices_per_device,
                            slice_memory_coeff=self.args.slice_memory_coeff,
                            data_splitting_policy=self.args.data_splitting_policy)
        if len(regions) > 1:
            raise UniversalBackprojectError('Region does not fit to the GPU memory')

        graph = Ufo.TaskGraph()
        if not (args.only_bp or dark is None or flat is None):
            ffc = get_task('flat-field-correct', processing_node=gpu)
            ffc.props.fix_nan_and_inf = self.args.fix_nan_and_inf
            ffc.props.absorption_correct = self.args.absorptivity
            first = ffc
        else:
            first = None

        (first, last) = setup_graph(self.args, graph, x_region, y_region, self.args.region,
                                    first, gpu=gpu, index=gpu_index, do_output=False)
        output_dims = 2
        if args.slice_metric:
            output_dims = 1
            if args.slice_metric == 'msag':
                measure_task = get_task('measure', processing_node=gpu, axis=-1, metric='sum')
                gradient_task = get_task('gradient', processing_node=gpu, direction='both_abs')
                calculate_task = get_task('calculate', processing_node=gpu, expression='-v')
                graph.connect_nodes(last, gradient_task)
                graph.connect_nodes(gradient_task, measure_task)
                graph.connect_nodes(measure_task, calculate_task)
            else:
                measure_task = get_task('measure', processing_node=gpu, axis=-1,
                                        metric=self.args.slice_metric)
                graph.connect_nodes(last, measure_task)
        elif args.only_bp:
            graph = first

        super(UniversalBackproject, self).__init__(graph, get_output=True, output_dims=output_dims,
                                                   scheduler=scheduler, copy_inputs=copy_inputs)

    @coroutine
    def __call__(self, consumer):
        def process_projection(projection, dark, flat):
            projection = projection[self.args.y:self.args.y + self.args.height]
            if projection.dtype != np.float32:
                projection = projection.astype(np.float32)
            self.insert(projection, index=0)
            if self.dark is not None and self.flat is not None:
                self.insert(dark, index=1)
                self.insert(flat, index=2)

        if not self._started:
            self.start()

        projection = yield
        st = time.time()
        process_projection(projection, self.dark, self.flat)

        i = 1
        while True:
            projection = yield
            i += 1
            if i == self.args.number:
                LOG.debug('Last projection came')
            process_projection(projection, None, None)
            if i == self.args.number:
                self.stop()
                if self.before_download_event:
                    LOG.debug('Waiting for event before download')
                    self.before_download_event.wait()
                LOG.debug('Backprojection duration: %.2f s', time.time() - st)
                st = time.time()
                for k in np.arange(*self.args.region):
                    result = self.result()[0]
                    consumer.send(result)
                LOG.debug('Volume downloaded in: %.2f s', time.time() - st)
                self.wait()


class UniversalBackprojectManager(object):
    def __init__(self, args, dark=None, flat=None, regions=None, copy_inputs=False,
                 projection_sleep_time=0 * q.s):
        self.regions = regions
        self.copy_inputs = copy_inputs
        self.projection_sleep_time = projection_sleep_time
        self.projections = None
        self._resources = []
        self.volume = None
        self.dark = dark
        self.flat = flat
        self.args = args
        self._consume_event = threading.Event()
        self._process_event = threading.Event()
        self._consume_event.set()
        self._process_event.set()
        self._update()

    def _update(self):
        """Update the regions and volume sizes based on changed args or region."""
        x_region, y_region, z_region = get_reconstruction_regions(self.args)
        if not self._resources:
            self._resources = [Ufo.Resources()]
        gpus = self._resources[0].get_gpu_nodes()
        if self.regions is None:
            self._regions = make_runs(gpus, x_region, y_region, z_region,
                                      DTYPE_CL_SIZE[self.args.store_type],
                                      slices_per_device=self.args.slices_per_device,
                                      slice_memory_coeff=self.args.slice_memory_coeff,
                                      data_splitting_policy=self.args.data_splitting_policy,
                                      num_gpu_threads=self.args.num_gpu_threads)
        else:
            self._regions = self.regions
        offset = 0
        for batch in self._regions:
            for i, region in batch:
                if len(self._resources) < len(batch):
                    self._resources.append(Ufo.Resources())
                offset += len(np.arange(*region))
        if self.args.slice_metric:
            shape = (offset,)
        else:
            shape = (offset, len(np.arange(*y_region)), len(np.arange(*x_region)))
        if self.volume is None or shape != self.volume.shape:
            self.join_consuming()
            self.volume = np.empty(shape, dtype=np.float32)

    def produce(self):
        sleep_time = self.projection_sleep_time.to(q.s).magnitude
        for i in range(self.args.number):
            while self._num_received_projections < i + 1:
                time.sleep(sleep_time)
            yield self.projections[i]

    @coroutine
    def consume(self, offset):
        i = 0
        while True:
            item = yield
            self.volume[offset + i] = item
            i += 1

    def join_processing(self):
        LOG.debug('Waiting for backprojectors to finish')
        self._process_event.wait()

    def join_consuming(self):
        LOG.debug('Waiting for volume processing to finish')
        self._consume_event.wait()

    def join(self):
        self.join_processing()
        self.join_consuming()

    def consume_volume(self, consumer):
        self.join_consuming()
        self._consume_event.clear()

        def send_volume():
            out_st = time.time()
            for s in self.volume:
                consumer.send(s)
            out_duration = time.time() - out_st
            out_size = self.volume.nbytes / 2. ** 20
            LOG.debug('Volume sending duration: %.2f s, speed: %.2f MB/s',
                      out_duration, out_size / out_duration)
            self._consume_event.set()

        threading.Thread(target=send_volume).start()

    def find_parameter(self, parameter, metric='msag', region=None, z=None,
                       method='powell', method_options=None, guess=None):
        orig_args = self.args
        self.args = copy.deepcopy(self.args)
        self.args.slice_metric = metric
        self.args.data_splitting_policy = 'one'
        self.args.z_parameter = parameter
        self.args.z = z or 0

        if region is None:
            from scipy.optimize import minimize

            def score(axis):
                axis = axis[0]
                LOG.info('Optimization axis position: %g', axis)
                self.args.region = [axis, axis + 1, 1.]
                inject(self.projections, self(block=True))
                return -self.volume[0]

            if guess is None:
                if parameter == 'center-position-x':
                    guess = self.args.width / 2.
                else:
                    guess = 0.
            res = minimize(score, guess, method=method, options=method_options)
            LOG.info('%s', res.message)
            result = float(res.x)
        else:
            self.args.region = region
            inject(self.projections, self(block=True))
            result = np.argmax(self.volume) * region[-1] + region[0]

        orig_args.center_position_x = [result]
        self.args = orig_args

        return result

    @coroutine
    def __call__(self, consumer=None, block=False, wait_for_events=None,
                 wait_for_projections=False):
        self.join_processing()
        self._process_event.clear()
        LOG.debug('Backprojector manager start')
        st = time.time()
        self._num_received_projections = 0
        aborted = False

        def prepare_and_start():
            """Make sure the arguments are up-to-date."""
            if wait_for_events is not None:
                LOG.debug('Waiting for events')
                for event in wait_for_events:
                    event.wait()
                LOG.debug('Waiting for events done (cached projections: %d)',
                          self._num_received_projections)

            self._update()

            def start_one(index):
                """Start one backprojector with a specific GPU ID in a separate thread."""
                offset = 0
                for i in range(self._batch_index):
                    for j, region in self._regions[i]:
                        offset += len(np.arange(*region))
                batch = self._regions[self._batch_index]
                offset += sum([len(np.arange(*reg)) for j, reg in batch[:index]])

                gpu_index, region = self._regions[self._batch_index][index]
                bp = UniversalBackproject(self.args,
                                          resources=self._resources[index],
                                          gpu_index=gpu_index,
                                          dark=self.dark,
                                          flat=self.flat,
                                          region=region,
                                          copy_inputs=self.copy_inputs,
                                          before_download_event=self._consume_event)
                inject(self.produce(), bp(self.consume(offset)))

            # Distribute work
            pool = ThreadPool(processes=len(self._resources))
            for i in range(len(self._regions)):
                if aborted:
                    break
                self._batch_index = i
                pool.map(start_one, range(len(self._regions[i])))
            pool.close()
            pool.join()

            if not aborted:
                # Process results
                duration = time.time() - st
                LOG.debug('Backprojectors duration: %.2f s', duration)
                in_size = self.projections.nbytes / 2. ** 20
                out_size = self.volume.nbytes / 2. ** 20
                LOG.debug('Input size: %g GB, output size: %g GB', in_size / 1024, out_size / 1024)
                LOG.debug('Performance: %.2f GUPS (In: %.2f MB/s, out: %.2f MB/s)',
                          self.volume.size * self.projections.shape[0] * 1e-9 / duration,
                          in_size / duration, out_size / duration)
                if consumer:
                    self.consume_volume(consumer)
            # Enable processing only after the consumer starts to make sure self.join()
            # works as expected
            self._process_event.set()

        if not wait_for_projections:
            arg_thread = threading.Thread(target=prepare_and_start)
            arg_thread.start()
        LOG.debug('Backprojectors initialization duration: %.2f ms', (time.time() - st) * 1000)

        try:
            projection = yield
            in_shape = (self.args.number, self.args.height, self.args.width)
            if (self.projections is None or in_shape != self.projections.shape or projection.dtype !=
                    self.projections.dtype):
                self.projections = np.empty(in_shape, dtype=projection.dtype)
            self.projections[0] = projection
            self._num_received_projections = 1

            finalize = True
            while True:
                projection = yield
                if self._num_received_projections < self.args.number:
                    self.projections[self._num_received_projections] = projection
                    self._num_received_projections += 1
                if finalize and self._num_received_projections == self.args.number:
                    finalize = False
                    if wait_for_projections:
                        arg_thread = threading.Thread(target=prepare_and_start)
                        arg_thread.start()
                    LOG.debug('Last projection dispatched by manager')
                    if block:
                        self.join()
        except GeneratorExit:
            if self._num_received_projections < self.projections.shape[0]:
                LOG.error('Not enough projections received (%d from %d)',
                          self._num_received_projections, self.args.number)
                aborted = True
                # Let UFO process fake projections until the graph can be aborted as well
                self._num_received_projections = self.projections.shape[0]
                arg_thread.join()


class UniversalBackprojectError(Exception):
    pass


class UniversalBackprojectArgsError(Exception):
    pass
