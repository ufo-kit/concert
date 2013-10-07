import threading
import numpy as np

try:
    from gi.repository import Ufo
    import ufonp
except ImportError:
    print("Ufo typelibs or ufonp are not installed")


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

    """Process to inject NumPy data into a Ufo processing graph.

    :class:`InjectProcess` can also be used as a context manager, in which
    case it will call :meth:`.run` on entering the manager and :meth:`.wait` on
    exiting it.

    *graph* must either be a Ufo.TaskGraph or a Ufo.TaskNode object.  If it is
    a graph the input task will be connected to the first root, otherwise a new
    graph will be created with the input task connecting to *graph*.
    """

    def __init__(self, graph):
        self.input_task = Ufo.InputTask()

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

        self.ufo_buffer = None

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wait()
        return True

    def run(self):
        """
        Run the processing in a new thread.

        Use :meth:`.push` to insert data into the processing chaing and
        :meth:`.wait` to wait until processing has finished."""
        def run_scheduler():
            sched = Ufo.Scheduler()
            sched.run(self.graph)

        self.thread = threading.Thread(target=run_scheduler)
        self.thread.start()

    def push(self, array):
        """
        Insert *array* into the processing chain.

        .. note:: *array* must be a NumPy compatible array.
        """
        if self.ufo_buffer is None:
            self.ufo_buffer = ufonp.fromarray(array.astype(np.float32))
        else:
            self.ufo_buffer = self.input_task.get_input_buffer()
            ufonp.fromarray_inplace(self.ufo_buffer, array.astype(np.float32))

        self.input_task.release_input_buffer(self.ufo_buffer)

    def consume(self):
        """Co-routine compatible consumer."""
        while True:
            item = yield
            self.push(item)

    def wait(self):
        """Wait until processing has finished."""
        self.input_task.stop()
        self.thread.join()
