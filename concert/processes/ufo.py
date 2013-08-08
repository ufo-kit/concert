"""
A graph-based data processing framework is part of the `Ufo project`_. In order
to make both projects as decoupled as possible, the :class:`.UfoProcess` class
let's an ordinary Ufo task graph look like it can be parameterized. Let's
assume, we have the following task description ::

    from gi.repository import Ufo

    # Prepare the graph
    pm = Ufo.PluginManager()
    reader = pm.get_task('reader')
    writer = pm.get_task('writer')
    bp = pm.get_task('backproject')
    bp.axis_pos = 0

    graph = Ufo.TaskGraph()
    graph.read_from_file('json')
    graph.connect_nodes(reader, bp)
    graph.connect_nodes(bp, writer)

To expose Ufo node properties as Concert parameters, we have to pass the graph,
the node and the node's property name to the :class:`UfoProcess` constructor.
In the following example, we make the axis position available to the control
system ::

    from concert.processes.ufo import UfoProcess

    process = UfoProcess(graph, bp, 'axis-pos')

Now, we can use ``process`` like any other regular device, for example print
the value or scan along a "trajectory"::

    from concert.processes.base import Scanner

    scanner = Scanner(process['axis-pos'], feedback)
    x, y = scanner.run().result()

.. _Ufo project: http://ufo.kit.edu
"""
import multiprocessing
from concert.base import Parameter
from concert.processes.base import Process
from concert.asynchronous import async


class UfoProcess(Process):

    """Wraps a Ufo task graph and export selected node properties.

    *graph* must be a Ufo task graph. *node* is a node that is connected inside
    *graph* *prop_name* is a property name of *node* and used to expose this
    property.

    :meth:`run` executes *graph* with its own scheduler instance. Use the
    *config* parameter to pass a UfoConfiguration to the scheduler.
    """

    def __init__(self, graph, node, prop_name, config=None):
        self._graph = graph
        self._config = config

        def _create_getter(node, param):
            def _wrapper():
                value = node.get_property(param)
                return value
            return _wrapper

        def _create_setter(node, param):
            def _wrapper(value):
                node.set_property(param, value)
                return self.run()
            return _wrapper

        def _create_parameter(node, prop):
            from gi.repository import GObject

            getter, setter = None, None

            if prop.flags & GObject.ParamFlags.READABLE:
                getter = _create_getter(node, prop.name)

            if prop.flags & GObject.ParamFlags.WRITABLE:
                setter = _create_setter(node, prop.name)

            param = Parameter(prop.name, getter, setter, doc=prop.blurb)
            return param

        prop = [p for p in node.props if p.name == prop_name]

        if prop:
            parameter = _create_parameter(node, prop[0])
            super(UfoProcess, self).__init__([parameter])
        else:
            msg = "Parameter {0} not in {1}".format(prop_name, node)
            raise ValueError(msg)

    @async
    def run(self):
        """Execute the graph."""
        def target():
            """Configure and run."""
            from gi.repository import Ufo

            if self._config:
                sched = Ufo.Scheduler(config=self._config)
            else:
                sched = Ufo.Scheduler()

            sched.run(self._graph)

        proc = multiprocessing.Process(target=target)
        proc.start()
        proc.join()
