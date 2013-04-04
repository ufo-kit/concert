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

To expose Ufo node properties as Concert parameters, we have to provide a map,
that translates from nodes to a list of property names. In the following
example, we want to make the axis position available to the control system ::

    from concert.processes.ufo import UfoProcess

    process = UfoProcess(graph, {bp: ['axis-pos']})

Now, we can use ``process`` like any other regular device, for example print
the value or scan along a "trajectory"::

    from concert.processes.scan import ascan

    def handle(parameters):
        print("Set point reached, inspect data")

    print(process['axis-pos'])
    ascan([(process['axis-pos'], 0, 1024)], 30, handle)

.. _Ufo project: http://ufo.kit.edu
"""
from concert.base import Device, Parameter
from concert.asynchronous import executor


class UfoProcess(Device):
    """Wraps a Ufo task graph and export selected node properties.

    *graph* must be a Ufo task graph. *node_map* is a dictionary that maps
    nodes that are connected within *graph* to its properties. Each property is
    then exposed as a parameter of the class.

    :meth:`run` executes *graph* with its own scheduler instance. Use the
    *config* parameter to pass a UfoConfiguration to the scheduler.
    """

    def __init__(self, graph, node_map, config=None):
        self._graph = graph
        self._config = config
        params = []

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

        for node, prop_names in node_map.items():
            for prop_name in prop_names:
                prop = [p for p in node.props if p.name == prop_name]
                if prop:
                    params.append(_create_parameter(node, prop[0]))
                else:
                    msg = "Parameter {0} not in {1}".format(prop_name, node)
                    raise ValueError(msg)

        super(UfoProcess, self).__init__(params)

    def run(self):
        """Execute the graph."""
        from gi.repository import Ufo

        if self._config:
            sched = Ufo.Scheduler(config=self._config)
        else:
            sched = Ufo.Scheduler()

        return executor.submit(sched.run, self._graph)
