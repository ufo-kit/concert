from __future__ import absolute_import
import threading
import numpy as np

try:
    from gi.repository import Ufo
    import ufo.numpy
except ImportError as e:
    print(str(e))

from concert.coroutines import coroutine


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
    a graph the input task will be connected to the first root, otherwise a new
    graph will be created with the input task connecting to *graph*.
    """

    def __init__(self, graph, get_output=False):
        self.input_task = Ufo.InputTask()
        self.output_task = None

        if isinstance(graph, Ufo.TaskGraph):
            self.graph = graph
            roots = self.graph.get_roots()
            self.graph.connect_nodes(self.input_task, roots[0])
        elif isinstance(graph, Ufo.TaskNode):
            self.graph = Ufo.TaskGraph()
            self.graph.connect_nodes(self.input_task, graph)
        else:
            msg = 'graph is neither Ufo.TaskGraph nor Ufo.TaskNode'
            raise ValueError(msg)

        if get_output:
            self.output_task = Ufo.OutputTask()
            leaves = self.graph.get_leaves()
            self.graph.connect_nodes(leaves[0], self.output_task)

        self.ufo_buffer = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wait()
        return True

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

    def insert(self, array):
        """
        Insert *array* into the processing chain.

        .. note:: *array* must be a NumPy compatible array.
        """
        if self.ufo_buffer is None:
            self.ufo_buffer = ufo.numpy.fromarray(array.astype(np.float32))
        else:
            self.ufo_buffer = self.input_task.get_input_buffer()
            ufo.numpy.fromarray_inplace(self.ufo_buffer, array.astype(np.float32))

        self.input_task.release_input_buffer(self.ufo_buffer)

    def result(self):
        if self.output_task:
            buf = self.output_task.get_output_buffer()
            result = np.copy(ufo.numpy.asarray(buf))
            self.output_task.release_output_buffer(buf)
            return result

    def consume(self):
        """Co-routine compatible consumer."""
        while True:
            item = yield
            self.insert(item)

    def wait(self):
        """Wait until processing has finished."""
        self.input_task.stop()
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
        self._started = False

        if axis_pos:
            self.backprojector.props.axis_pos = axis_pos

        graph = Ufo.TaskGraph()
        graph.connect_nodes(self.fft, self.fltr)
        graph.connect_nodes(self.fltr, self.ifft)
        graph.connect_nodes(self.ifft, self.backprojector)

        super(Backproject, self).__init__(graph, get_output=True)

    @coroutine
    def __call__(self, consumer):
        """Get a sinogram, do filtered backprojection and send it to *consumer*."""
        if not self._started:
            self.start()
            self._started = True

        slice = None

        while True:
            sinogram = yield

            if slice is None:
                width = sinogram.shape[1]
                slice = np.empty((width, width), dtype=np.float32)

            self.insert(sinogram)
            slice = self.result()[:width, :width]

            consumer.send(slice)
