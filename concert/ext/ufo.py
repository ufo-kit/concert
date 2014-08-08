from __future__ import absolute_import
import threading
import numpy as np

try:
    from gi.repository import Ufo
    import ufo.numpy
except ImportError as e:
    print(str(e))

from concert.coroutines.base import coroutine


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
    and :meth:`.wait` on exiting it.

    *graph* must either be a Ufo.TaskGraph or a Ufo.TaskNode object.  If it is
    a graph the input tasks will be connected to the roots, otherwise a new
    graph will be created.
    """

    def __init__(self, graph, get_output=False):
        self.output_task = None
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
            self.output_task = Ufo.OutputTask()
            leaves = self.graph.get_leaves()
            self.graph.connect_nodes(leaves[0], self.output_task)

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
            consumer.send(self.result())

    def start(self):
        """
        Run the processing in a new thread.

        Use :meth:`.push` to insert data into the processing chaing and
        :meth:`.wait` to wait until processing has finished."""
        def run_scheduler():
            sched = Ufo.Scheduler()
            sched.run(self.graph)

        self.thread = threading.Thread(target=run_scheduler)
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
            self.ufo_buffers[node][index] = ufo.numpy.fromarray(array.astype(np.float32))
        else:
            self.ufo_buffers[node][index] = self.input_tasks[node][index].get_input_buffer()
            ufo.numpy.fromarray_inplace(self.ufo_buffers[node][index], array.astype(np.float32))

        self.input_tasks[node][index].release_input_buffer(self.ufo_buffers[node][index])

    def result(self):
        if self.output_task:
            buf = self.output_task.get_output_buffer()
            result = np.copy(ufo.numpy.asarray(buf))
            self.output_task.release_output_buffer(buf)
            return result

    def wait(self):
        """Wait until processing has finished."""
        for input_tasks in self.input_tasks.values():
            for input_task in input_tasks:
                input_task.stop()
        self.thread.join()


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
        self.fft = self.pm.get_task('fft', dimensions=1)
        self.ifft = self.pm.get_task('ifft', dimensions=1)
        self.fltr = self.pm.get_task('filter')
        self.backprojector = self.pm.get_task('backproject')

        if axis_pos:
            self.backprojector.props.axis_pos = axis_pos

        graph = Ufo.TaskGraph()
        graph.connect_nodes(self.fft, self.fltr)
        graph.connect_nodes(self.fltr, self.ifft)
        graph.connect_nodes(self.ifft, self.backprojector)

        super(Backproject, self).__init__(graph, get_output=True)

        self.output_task.props.num_dims = 2

    @property
    def axis_position(self):
        return self.backprojector.props.axis_pos

    @axis_position.setter
    def axis_position(self, position):
        self.backprojector.props.axis_pos = position

    @coroutine
    def __call__(self, consumer):
        """Get a sinogram, do filtered backprojection and send it to *consumer*."""
        def process(sino):
            self.insert(sino)
            consumer.send(self.result())

        if not self._started:
            self.start()

        sinogram = yield
        self.ifft.props.crop_width = sinogram.shape[1]
        process(sinogram)

        while True:
            sinogram = yield
            process(sinogram)


class FlatCorrectedBackproject(InjectProcess):

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
        self.sino_correction = self.pm.get_task('sino-correction')
        self.fft = self.pm.get_task('fft', dimensions=1)
        self.ifft = self.pm.get_task('ifft', dimensions=1)
        self.fltr = self.pm.get_task('filter')
        self.backprojector = self.pm.get_task('backproject')

        if axis_pos:
            self.backprojector.props.axis_pos = axis_pos

        graph = Ufo.TaskGraph()
        graph.connect_nodes(self.sino_correction, self.fft)
        graph.connect_nodes(self.fft, self.fltr)
        graph.connect_nodes(self.fltr, self.ifft)
        graph.connect_nodes(self.ifft, self.backprojector)

        super(FlatCorrectedBackproject, self).__init__(graph, get_output=True)

        self.flat_row = flat_row
        self.dark_row = dark_row

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
            self.ifft.props.crop_width = row.shape[0]

        self._dark_row = row

    @property
    def flat_row(self):
        return self._flat_row

    @flat_row.setter
    def flat_row(self, row):
        if row is not None:
            row = row.astype(np.float32)

        self._flat_row = row

    @coroutine
    def __call__(self, consumer):
        """Get a sinogram, do filtered backprojection and send it to *consumer*."""
        if not self._started:
            self.start()

        while True:
            sinogram = yield
            self.insert(sinogram.astype(np.float32), node=self.sino_correction, index=0)
            if self.dark_row is None or self.flat_row is None:
                raise ValueError('Both flat and dark rows must be set')
            self.insert(self.dark_row, node=self.sino_correction, index=1)
            self.insert(self.flat_row, node=self.sino_correction, index=2)
            consumer.send(self.result())
