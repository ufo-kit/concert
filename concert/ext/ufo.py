
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
    from tofu.config import SECTIONS, GEN_RECO_PARAMS
    from tofu.util import setup_padding, get_reconstruction_regions
    from tofu.genreco import (CTGeometry, setup_graph, set_projection_filter_scale, make_runs,
                              DTYPE_CL_SIZE)
    from tofu.tasks import get_task
except ImportError:
    print("You must install tofu to use Ufo features, see "\
                         "'https://github.com/ufo-kit/tofu.git'", file=sys.stderr)

from multiprocessing.pool import ThreadPool
from concert.casync import casync
from concert.imageprocessing import filter_low_frequencies
from concert.quantities import q
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

    async def __call__(self, producer):
        """Co-routine compatible consumer."""
        if not self._started:
            self.start()

        async for item in producer:
            self.insert(item)
            yield self.result(leave_index=0)

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
                node = list(self.input_tasks.keys())[0]
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
            indices = list(range(len(self.output_tasks))) if leave_index is None else [leave_index]
            results = []
            for index in indices:
                buf = self.output_tasks[index].get_output_buffer()
                if not buf:
                    return [None]
                results.append(np.copy(ufo.numpy.asarray(buf)))
                self.output_tasks[index].release_output_buffer(buf)

            if leave_index is not None:
                results = results[0]

            return results

    def stop(self):
        """Stop input tasks."""
        for input_tasks in list(self.input_tasks.values()):
            for input_task in input_tasks:
                input_task.stop()

    def wait(self):
        """Wait until processing has finished."""
        self.stop()
        self.thread.join()
        self._started = False


class FlatCorrect(InjectProcess):
    """
    Flat-field correction.
    """
    def __init__(self, dark, flat, absorptivity=True, fix_nan_and_inf=True, copy_inputs=False):
        self.dark = dark
        self.flat = flat
        self.ffc = get_task('flat-field-correct')
        self.ffc.props.fix_nan_and_inf = fix_nan_and_inf
        self.ffc.props.absorption_correct = absorptivity
        super(FlatCorrect, self).__init__(self.ffc, get_output=True, output_dims=2,
                                          copy_inputs=copy_inputs)

    async def __call__(self, producer):
        """Co-routine compatible consumer."""
        def process_one(projection, dark, flat):
            if projection.dtype != np.float32:
                projection = projection.astype(np.float32)
            self.insert(projection, index=0)
            self.insert(dark, index=1)
            self.insert(flat, index=2)
            consumer.send(self.result(leave_index=0))

        if not self._started:
            self.start()

        fc = True
        async for projection in producer:
            if fc:
                yield process_one(projection, self.dark.astype(np.float32), self.flat.astype(np.float32))
                fc = False
            else:
                yield process_one(projection, None, None)


class GeneralBackprojectArgs(object):
    def __init__(self, center_position_x, center_position_z, number,
                 overall_angle=np.pi):
        self._slice_metric = None
        self._slice_metrics = ['min', 'max', 'sum', 'mean', 'var', 'std', 'skew',
                               'kurtosis', 'sag']
        self._z_parameters = SECTIONS['general-reconstruction']['z-parameter']['choices']
        for section in GEN_RECO_PARAMS:
            for arg in SECTIONS[section]:
                settings = SECTIONS[section][arg]
                default = settings['default']
                if default is not None and 'type' in settings:
                    default = settings['type'](default)
                setattr(self, arg.replace('-', '_'), default)
        self.y = 0
        self.height = None
        self.width = None
        self.center_position_x = center_position_x
        self.center_position_z = center_position_z
        self.number = number
        self.overall_angle = overall_angle

    @property
    def z_parameters(self):
        return self._z_parameters

    @property
    def slice_metrics(self):
        return self._slice_metrics

    @property
    def slice_metric(self):
        return self._slice_metric

    @slice_metric.setter
    def slice_metric(self, metric):
        if metric not in [None] + self.slice_metrics:
            raise GeneralBackprojectArgsError("Metric '{}' not known".format(metric))
        self._slice_metric = metric

    @property
    def z_parameter(self):
        return self._z_parameter

    @z_parameter.setter
    def z_parameter(self, name):
        if name not in self.z_parameters:
            raise GeneralBackprojectArgsError("Unknown z parameter '{}'".format(name))
        self._z_parameter = name


class GeneralBackproject(InjectProcess):
    def __init__(self, args, resources=None, gpu_index=0, flat=None, dark=None, region=None,
                 copy_inputs=False, before_download_event=None):
        if args.width is None or args.height is None:
            raise GeneralBackprojectError('width and height must be set in GeneralBackprojectArgs')
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
        if not self.args.disable_projection_crop:
            geometry.optimize_args()
        self.args = geometry.args
        self.dark = dark
        self.flat = flat
        if self.dark is not None and self.flat is not None:
            LOG.debug('Flat correction on')
            self.dark = self.dark[self.args.y:self.args.y + self.args.height].astype(np.float32)
            self.flat = self.flat[self.args.y:self.args.y + self.args.height].astype(np.float32)

        regions = make_runs([gpu], [gpu_index], x_region, y_region, self.args.region,
                            DTYPE_CL_SIZE[self.args.store_type],
                            slices_per_device=self.args.slices_per_device,
                            slice_memory_coeff=self.args.slice_memory_coeff,
                            data_splitting_policy=self.args.data_splitting_policy)
        if len(regions) > 1:
            raise GeneralBackprojectError('Region does not fit to the GPU memory')

        graph = Ufo.TaskGraph()
        if dark is not None and flat is not None:
            ffc = get_task('flat-field-correct', processing_node=gpu)
            ffc.props.fix_nan_and_inf = self.args.fix_nan_and_inf
            ffc.props.absorption_correct = self.args.absorptivity
            first = ffc
        else:
            first = None

        (first, last) = setup_graph(self.args, graph, x_region, y_region, self.args.region,
                                    source=first, gpu=gpu, index=gpu_index, do_output=False,
                                    make_reader=False)
        output_dims = 2
        if args.slice_metric:
            output_dims = 1
            metric = self.args.slice_metric
            if args.slice_metric == 'sag':
                metric = 'sum'
                gradient_task = get_task('gradient', processing_node=gpu, direction='both_abs')
                graph.connect_nodes(last, gradient_task)
                last = gradient_task
            measure_task = get_task('measure', processing_node=gpu, axis=-1, metric=metric)
            graph.connect_nodes(last, measure_task)
        elif first == last:
            # There are no other processing steps other than back projection
            LOG.debug('Only back projection, no other processing')
            graph = first

        super(GeneralBackproject, self).__init__(graph, get_output=True, output_dims=output_dims,
                                                 scheduler=scheduler, copy_inputs=copy_inputs)

    async def __call__(self, producer):
        def process_projection(projection, dark, flat):
            if projection is None:
                return False
            elif not self._started:
                self.start()
            projection = projection[self.args.y:self.args.y + self.args.height]
            if projection.dtype != np.float32:
                projection = projection.astype(np.float32)
            self.insert(projection, index=0)
            if self.dark is not None and self.flat is not None:
                self.insert(dark, index=1)
                self.insert(flat, index=2)

            return True

        def consume_volume():
            if not self._started:
                yield None
                return
            self.stop()
            if self.before_download_event:
                LOG.debug('Waiting for event before download')
                self.before_download_event.wait()
            st = time.time()
            for k in np.arange(*self.args.region):
                result = self.result()[0]
                if result is None:
                    LOG.warn('Not all slices received (last: %g)', k)
                    break
                yield result
            LOG.debug('Volume downloaded in: %.2f s', time.time() - st)
            self.wait()

        i = 0
        async for projection in producer:
            if i == 0:
                st = time.time()
            i += 1
            processed = process_projection(projection,
                                           self.dark if i == 1 else None,
                                           self.flat if i == 1 else None)
            if not processed or i == self.args.number:
                LOG.debug('Backprojected %d projections, duration: %.2f s', i, time.time() - st)
                consume_volume()


class GeneralBackprojectManager(object):
    def __init__(self, args, dark=None, flat=None, regions=None, copy_inputs=False,
                 projection_sleep_time=0 * q.s):
        self._aborted = False
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
        self._num_received_projections = 0
        self._num_processed_projections = 0

    @property
    def num_received_projections(self):
        return self._num_received_projections

    @property
    def num_processed_projections(self):
        return self._num_processed_projections

    def _update(self):
        """Update the regions and volume sizes based on changed args or region."""
        st = time.time()
        x_region, y_region, z_region = get_reconstruction_regions(self.args)
        if not self._resources:
            self._resources = [Ufo.Resources()]
        gpus = np.array(self._resources[0].get_gpu_nodes())
        gpu_indices = np.array(self.args.gpus or list(range(len(gpus))))
        if min(gpu_indices) < 0 or max(gpu_indices) > len(gpus) - 1:
            raise ValueError('--gpus contains invalid indices')
        gpus = gpus[gpu_indices]
        if self.regions is None:
            self._regions = make_runs(gpus, gpu_indices, x_region, y_region, z_region,
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
        LOG.debug('Backprojector manager update duration: %g s', time.time() - st)

    def produce(self):
        sleep_time = self.projection_sleep_time.to(q.s).magnitude
        for i in range(self.args.number):
            while self._num_received_projections < i + 1:
                if self._aborted:
                    break
                time.sleep(sleep_time)
            if self._aborted:
                yield None
                break
            yield self.projections[i]
            self._num_processed_projections = i + 1

    async def consume(self, offset, producer):
        i = 0
        async for item in producer:
            if item is None:
                self.abort()
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
            out_size = self.volume.nbytes / 2 ** 20
            LOG.debug('Volume sending duration: %.2f s, speed: %.2f MB/s',
                      out_duration, out_size / out_duration)
            self._consume_event.set()

        threading.Thread(target=send_volume).start()

    @casync
    def find_parameters(self, parameters, projections=None, metrics=('sag',), regions=None,
                        iterations=1, fwhm=0, minimize=(True,), z=None, method='powell',
                        method_options=None, guesses=None, bounds=None, store=True):
        """Find reconstruction parameters. *parameters* (see
        :attr:`.GeneralBackprojectArgs.z_parameters`) are the names of the parameters which should
        be found, *projections* are the input data and if not specified, the ones from last
        reconstruction are used. *z* specifies the height in which the parameter is looked for. If
        *store* is True, the found parameter values are stored in the reconstruction arguments.
        Optimization is done either brute-force if *regions* are not specified or one of the scipy
        minimization methods is used, see below.

        If *regions* are specified, they are reconstructed for the corresponding parameters and a
        metric from *metrics* list is applied. Thus, first parameter in *parameters* is
        reconstructed within the first region in *regions* and the first metric (see
        :attr:`.GeneralBackprojectArgs.slice_metrics`) in *metrics* is applied and so on. If
        *metrics* is of length 1 then it is applied to all parameters. *minimize* is a tuple
        specifying whether each parameter in the list should be minimized (True) or maximized
        (False). After every parameter is processed, the parameter optimization result is stored and
        the next parameter is optimized in such a way, that the result of the optimization of the
        previous parameter already takes place. *iterations* specifies how many times are all the
        parameters reconstructed. *fwhm* specifies the full width half maximum of the gaussian
        window used to filter out the low frequencies in the metric, which is useful when the region
        for a metric is large. If the *fwhm* is specified, the region must be at least 4 * fwhm
        large. If *fwhm* is 0 no filtering is done.

        If *regions* is not specified, :func:`scipy.minimize` is used to find the parameter, where
        the optimization method is given by the *method* parameter, *method_options* are passed as
        *options* to the minimize function and *guesses* are initial guesses in the order of the
        *parameters* list. If *bounds* are given, they represent the domains where to look for
        parameters, they are (min, max) tuples, also in the order of the *parameters* list. See
        documentation of :func:`scipy.minimize` for the list of minimization methods which support
        bounds specification. In this approach only the first in *metrics* is taken into account
        because the optimization happens on all parameters simultaneously, the same holds for
        *minimize*.
        """
        self.join_processing()
        if projections is None:
            if self.projections is None:
                raise GeneralBackprojectManagerError('*projections* must be specified if no '
                                                     ' reconstructions have been done yet')
            projections = self.projections
        orig_args = self.args
        self.args = copy.deepcopy(self.args)

        if regions is None:
            # No region specified, do a real optimization on the parameters vector
            from scipy import optimize

            def score(vector):
                for (parameter, value) in zip(parameters, vector):
                    setattr(self.args, parameter.replace('-', '_'), [value])
                inject(projections, self(block=True))
                result = sgn * self.volume[0]
                LOG.info('Optimization vector: %s, result: %g', vector, result)

                return result

            self.args.z_parameter = 'z'
            z = z or 0
            self.args.region = [z, z + 1, 1.]
            self.args.slice_metric = metrics[0]
            sgn = 1 if minimize[0] else -1
            if guesses is None:
                guesses = []
                for parameter in parameters:
                    if parameter == 'center-position-x':
                        guesses.append(self.args.width / 2)
                    else:
                        guesses.append(0.)
            LOG.info('Guesses: %s', guesses)
            result = optimize.minimize(score, guesses, method=method, bounds=bounds,
                                       options=method_options)
            LOG.info('%s', result.message)
            result = result.x
        else:
            # Regions specified, reconstruct given regions for given parameters and simply search
            # for extrema of the given metrics
            self.args.z = z or 0
            if fwhm:
                for region in regions:
                    if len(np.arange(*region)) < 4 * fwhm:
                        raise ValueError('All regions must be at least 4 * fwhm large '
                                         'when fwhm is specified')
            result = []
            if len(metrics) == 1:
                metrics = metrics * len(parameters)
            if len(minimize) == 1:
                minimize = minimize * len(parameters)
            for i in range(iterations):
                for (parameter, region, metric, minim) in zip(parameters, regions,
                                                              metrics, minimize):
                    self.args.slice_metric = metric
                    self.args.z_parameter = parameter
                    self.args.region = region
                    inject(projections, self(block=True))
                    sgn = 1 if minim else -1
                    values = self.volume
                    if fwhm:
                        values = filter_low_frequencies(values, fwhm=fwhm)[2 * int(fwhm):
                                                                           -2 * int(fwhm)]
                    param_result = (np.argmin(sgn * values) + 2 * fwhm) * region[2] + region[0]
                    setattr(self.args, parameter.replace('-', '_'), [param_result])
                    if i == iterations - 1:
                        result.append(param_result)
                    LOG.info('Optimizing %s, region: %s, metric: %s, minimize: %s, result: %g',
                             parameter, region, metric, minim, param_result)

        LOG.info('Optimization result: %s', result)

        if store:
            for (parameter, value) in zip(parameters, result):
                setattr(orig_args, parameter.replace('-', '_'), [value])
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
        self._num_processed_projections = 0
        self._aborted = False

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
                bp = GeneralBackproject(self.args,
                                        resources=self._resources[index],
                                        gpu_index=gpu_index,
                                        dark=self.dark,
                                        flat=self.flat,
                                        region=region,
                                        copy_inputs=self.copy_inputs,
                                        before_download_event=self._consume_event)
                self.consume(offset, bp(self.produce()))

            # Distribute work
            pool = ThreadPool(processes=len(self._resources))
            for i in range(len(self._regions)):
                if self._aborted:
                    break
                self._batch_index = i
                pool.map(start_one, list(range(len(self._regions[i]))))
            pool.close()
            pool.join()

            if not self._aborted:
                # Process results
                duration = time.time() - st
                LOG.debug('Backprojectors duration: %.2f s', duration)
                in_size = self.projections.nbytes / 2 ** 20
                out_size = self.volume.nbytes / 2 ** 20
                LOG.debug('Input size: %g GB, output size: %g GB', in_size / 1024, out_size / 1024)
                LOG.debug('Performance: %.2f GUPS (In: %.2f MB/s, out: %.2f MB/s)',
                          self.volume.size * self.projections.shape[0] * 1e-9 / duration,
                          in_size / duration, out_size / duration)
                if consumer:
                    self.consume_volume(consumer)
            # Enable processing only after the consumer starts to make sure self.join()
            # works as expected
            self._process_event.set()

        arg_thread = None
        try:
            projection = yield
            (self.args.height, self.args.width) = projection.shape
            if not wait_for_projections:
                arg_thread = threading.Thread(target=prepare_and_start)
                arg_thread.start()
            LOG.debug('Backprojectors initialization duration: %.2f ms', (time.time() - st) * 1000)
            in_shape = (self.args.number,) + projection.shape
            if (self.projections is None or in_shape != self.projections.shape or
                    projection.dtype != self.projections.dtype):
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
            if (self.projections is None or
                    self._num_received_projections < self.projections.shape[0]):
                self._aborted = True
                LOG.error('Not enough projections received (%d from %d)',
                          self._num_received_projections, self.args.number)
                if arg_thread:
                    arg_thread.join()

    def abort(self):
        self._aborted = True


class GeneralBackprojectManagerError(Exception):
    pass


class GeneralBackprojectError(Exception):
    pass


class GeneralBackprojectArgsError(Exception):
    pass
