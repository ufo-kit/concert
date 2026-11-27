"""Shared configuration for Cartesian scan experiments and directors."""

from math import prod
from collections.abc import Mapping
import re
from types import MappingProxyType

import numpy as np

from concert.base import Parameter, Parameterizable, ParameterValue, Selection, StateError


_EDITABLE_STATES = ['standby', 'error', 'cancelled']


def scan_index(shape, iteration, order='raster'):
    """Return grid indices for an iteration without materializing the grid.

    Raster and reflected snake traversal support any dimension. Spiral is a
    clockwise, outside-in traversal of a two-dimensional grid, starting at
    (0, 0) and moving along the second axis first. Orders use list indices,
    not the numerical values or distances between positions.

    :param shape: Non-empty sequence of positive axis lengths, in axis order.
    :param iteration: Zero-based acquisition number, smaller than the product
        of the axis lengths.
    :param order: ``'raster'``, ``'snake'`` or ``'spiral'``.
    :return: Tuple of zero-based indices, one per axis.
    :raises ValueError: For invalid shape/order or a non-2D spiral.
    :raises IndexError: If *iteration* is outside the grid.
    """
    if order not in ('raster', 'snake', 'spiral'):
        raise ValueError(f"Unknown scan order: {order}")
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("Scan axes must be non-empty")
    if order == 'spiral' and len(shape) != 2:
        raise ValueError("Spiral order requires exactly two axes")
    if not 0 <= iteration < prod(shape):
        raise IndexError("Scan iteration out of range")
    if order in ('raster', 'snake'):
        indices = []
        remaining = prod(shape)
        for size in shape:
            remaining //= size
            index, iteration = divmod(iteration, remaining)
            indices.append(index)
            if order == 'snake' and index % 2:
                iteration = remaining - 1 - iteration
        return tuple(indices)

    rows, columns = shape
    layer = 0
    while True:
        if rows == 1:
            return layer, layer + iteration
        if columns == 1:
            return layer + iteration, layer
        perimeter = 2 * (rows + columns) - 4
        if iteration >= perimeter:
            iteration -= perimeter
            rows, columns = rows - 2, columns - 2
            layer += 1
            continue
        if iteration < columns:
            return layer, layer + iteration
        iteration -= columns
        if iteration < rows - 1:
            return layer + 1 + iteration, layer + columns - 1
        iteration -= rows - 1
        if iteration < columns - 1:
            return layer + rows - 1, layer + columns - 2 - iteration
        iteration -= columns - 1
        return layer + rows - 2 - iteration, layer


class ScanAxis(Parameterizable):
    """A device parameter and its ordered positions.

    Inspect the usual Concert parameter table, use :meth:`set_values` for
    arbitrary positions, or :meth:`set_range` for a regular range. Axes cannot
    be changed while their owning experiment or director is running or paused.

    ``parameter`` is the read-only bound device parameter; ``values`` is an
    editable tuple of scalar positions. ``start``, ``stop`` and ``step`` are
    read-only recorded range settings (None for explicit lists). ``num_points``
    is the current list length. Editing configuration does not move the device.
    """

    parameter = Parameter(help="Device parameter being scanned")
    values = Parameter(help="Ordered scan positions")
    start = Parameter(help="Range start (None for explicit positions)")
    stop = Parameter(help="Range stop (None for explicit positions)")
    step = Parameter(help="Range step (None for explicit positions)")
    num_points = Parameter(help="Number of positions on this axis")

    async def __ainit__(self, parameter, values, *, owner=None, **kwargs):
        if not isinstance(parameter, ParameterValue):
            raise TypeError("Each scan parameter must be a bound device parameter")
        if not parameter.writable:
            raise ValueError(f"Parameter '{parameter.name}' is not writable")
        self._parameter = parameter
        self._owner = owner
        self._range = (None, None, None)
        await super().__ainit__(**kwargs)
        await self.set_values(values)

    async def _ensure_editable(self):
        if self._owner is not None and await self._owner.get_state() not in _EDITABLE_STATES:
            raise StateError("Cannot change scan axes while the scan is running or paused")

    @staticmethod
    def _validate_values(values):
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError("Scan values must be a non-empty sequence") from exc
        if not values:
            raise ValueError("Scan values must be a non-empty sequence")
        if any(np.ndim(value) != 0 for value in values):
            raise ValueError("Each scan position must be a scalar")
        return values

    async def _get_parameter(self):
        return self._parameter

    async def _get_values(self):
        return self._values

    async def _set_values(self, values):
        await self._ensure_editable()
        self._values = self._validate_values(values)
        self._range = (None, None, None)

    async def _get_start(self):
        return self._range[0]

    async def _get_stop(self):
        return self._range[1]

    async def _get_step(self):
        return self._range[2]

    async def _get_num_points(self):
        return len(self._values)

    async def set_range(self, start, stop, step, include_last=True):
        """Atomically replace positions with a range, including *stop* by default.

        Compatible quantities may use different units. Descending ranges need a
        negative step. Like :func:`concert.processes.common.ascan`, the exact
        endpoint is appended when *include_last* is True.

        :param start: First position, a scalar or quantity.
        :param stop: Range endpoint, compatible with *start*.
        :param step: Nonzero increment pointing towards *stop*.
        :param include_last: Include the exact endpoint, even when the step
            does not divide the range evenly. Defaults to True.
        :raises ValueError: For non-finite arguments, invalid step or an empty
            resulting sequence. Previous configuration remains unchanged.
        :raises StateError: If the owning scan is running or paused.
        """
        await self._ensure_editable()
        units = getattr(start, 'units', None)
        if units is not None:
            stop, step = stop.to(units), step.to(units)
            first, last, stride = start.magnitude, stop.magnitude, step.magnitude
        else:
            first, last, stride = start, stop, step
        if not all(np.ndim(value) == 0 and np.isfinite(value)
                   for value in (first, last, stride)):
            raise ValueError("Range start, stop and step must be finite scalars")
        if stride == 0 or (last - first) * stride < 0:
            raise ValueError("Step must be nonzero and point towards stop")
        values = np.arange(first, last, stride)
        if include_last:
            values = np.concatenate((values, [last]))
        if units is not None:
            values = values * units
        self._values = self._validate_values(values)
        self._range = (start, stop, step)


class ScanMixin(Parameterizable):
    """Named scan axes, validation and plan display shared by scan runnables.

    Prefer ``axes={name: (parameter, values), ...}`` to keep each axis name,
    binding and positions together. Mapping insertion order determines axis
    order. A mapping can also be supplied as the first positional argument of
    the concrete scan constructors. Do not combine it with the legacy
    ``parameters``, ``values`` or ``names`` arguments.

    Used by local/remote scan experiments and the scan director. For each axis
    named ``x``, installs live Concert parameters ``scan_x`` (summary),
    ``scan_x_values`` (editable), and read-only ``scan_x_start``, ``scan_x_stop``,
    ``scan_x_step`` and ``scan_x_num_points``. All edits are restricted to
    standby, error or cancelled states. Names are sanitized to ASCII identifiers;
    generated name collisions raise ValueError.

    Shared parameters are ``scan_plan`` (table string), ``num_positions``
    (product of axis lengths), ``order`` and ``go_back``. The latter two are
    editable between runs. Subclasses implement acquisition and restoration.
    """

    scan_plan = Parameter(help="Scan order, axes, ranges and arbitrary positions")
    order = Selection(['raster', 'snake', 'spiral'], help="Traversal order of the scan grid")
    num_positions = Parameter(help="Total number of Cartesian positions per run")
    go_back = Parameter(help="Restore scanned parameters after the scan")

    async def __ainit__(self, *, parameters=None, values=None, axes=None, names=None,
                        go_back=False, order='raster', **kwargs):
        if isinstance(parameters, Mapping):
            if axes is not None:
                raise ValueError("Provide the axis mapping only once")
            axes, parameters = parameters, None
        if axes is not None:
            if parameters is not None or values is not None or names is not None:
                raise ValueError("Use either axes or parameters/values/names, not both")
            if not isinstance(axes, Mapping) or not axes:
                raise ValueError("axes must be a non-empty mapping of names to (parameter, values)")
            names, parameters, values = [], [], []
            for name, specification in axes.items():
                if not isinstance(specification, (tuple, list)) or len(specification) != 2:
                    raise ValueError(f"Axis '{name}' must be a (parameter, values) pair")
                parameter, positions = specification
                names.append(name)
                parameters.append(parameter)
                values.append(positions)
        elif parameters is None or values is None:
            raise ValueError("Provide axes or both parameters and values")
        if isinstance(parameters, ParameterValue):
            parameters, values = [parameters], [values]
        else:
            parameters, values = list(parameters), list(values)
        if not parameters or len(parameters) != len(values):
            raise ValueError("Provide one non-empty position sequence per parameter")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("A parameter cannot be scanned on multiple axes")
        names = list(names) if names is not None else [f'axis_{i}' for i in range(len(parameters))]
        if (len(names) != len(parameters)
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Provide one unique, non-empty name per axis")
        self._go_back = bool(go_back)
        self._axes = {}
        await super().__ainit__(**kwargs)
        for name, parameter, positions in zip(names, parameters, values):
            self._axes[name] = await ScanAxis(parameter, positions, owner=self)
        await self.set_order(order)
        self._install_axis_parameters()

    def _install_axis_parameters(self):
        """Expose live axis configuration as instance-specific Concert parameters."""
        parameters = {}
        mapping = {}
        for name, axis in self.axes.items():
            # Keep arbitrary display names while providing valid attribute names.
            prefix = 'scan_' + re.sub(r'[^\w]', '_', name, flags=re.ASCII)
            fields = {'': 'summary', '_values': 'values', '_start': 'start',
                      '_stop': 'stop', '_step': 'step', '_num_points': 'num_points'}
            mapping[name] = {}
            for suffix, field in fields.items():
                parameter_name = prefix + suffix
                if (parameter_name in parameters or parameter_name in self
                        or hasattr(self, parameter_name)
                        or hasattr(self, 'get_' + parameter_name)
                        or hasattr(self, 'set_' + parameter_name)):
                    raise ValueError(f"Scan axis names produce conflicting parameter '{parameter_name}'")

                async def getter(instance, axis=axis, field=field):
                    if field == 'summary':
                        return await instance._format_axis(axis)
                    return await getattr(axis, 'get_' + field)()

                async def setter(instance, values, axis=axis):
                    await axis.set_values(values)

                parameters[parameter_name] = Parameter(
                    fget=getter, fset=setter if field == 'values' else None,
                    help=f"Scan axis '{name}': {field}")
                mapping[name][field] = parameter_name

        # install_parameters adds methods to the current class and rebuilds it.
        # Isolate those methods and retain the original class in the MRO so
        # zero-argument super() in experiments/directors continues to work.
        self.__class__ = type(self.__class__.__name__, (self.__class__,), {})
        self.install_parameters(parameters)
        self._axis_parameter_names = mapping

    @property
    def axis_parameter_names(self):
        """Read-only mapping of axis names to generated parameter names.

        Field keys are ``summary``, ``values``, ``start``, ``stop``, ``step`` and
        ``num_points``. For example ``axis_parameter_names['x']['values']`` is
        ``'scan_x_values'``. Original axis names remain unchanged in ``axes``.
        """
        return MappingProxyType({name: MappingProxyType(fields)
                                 for name, fields in self._axis_parameter_names.items()})

    async def _format_axis(self, axis):
        parameter = await axis.get_parameter()
        values = await axis.get_values()
        start, stop, step = await axis.get_start(), await axis.get_stop(), await axis.get_step()
        if start is None:
            description = 'values=[' + ', '.join(str(value) for value in values) + ']'
        else:
            description = f'start={start}, stop={stop}, step={step}'
        return f'{parameter.name}: {description}; {len(values)} points; order={await self.get_order()}'

    @property
    def axes(self):
        """Named axes in scan order; edit their positions, not this mapping."""
        return MappingProxyType(self._axes)

    async def _get_num_positions(self):
        return prod([await axis.get_num_points() for axis in self.axes.values()])

    async def _get_order(self):
        return self._order

    async def _set_order(self, order):
        if await self.get_state() not in _EDITABLE_STATES:
            raise StateError("Cannot change scan order while the scan is running or paused")
        if order == 'spiral' and len(self.axes) != 2:
            raise ValueError("Spiral order requires exactly two axes")
        self._order = order

    async def get_position_indices(self, iteration):
        """Return the axis list indices visited at *iteration* in the chosen order.

        The returned tuple follows the insertion order of ``axes``. Indices
        identify configured list entries, including repeated values, rather
        than read-back device positions. Raises IndexError outside the grid.
        """
        shape = tuple([await axis.get_num_points() for axis in self.axes.values()])
        return scan_index(shape, iteration, await self.get_order())

    async def _get_go_back(self):
        return self._go_back

    async def _set_go_back(self, value):
        if await self.get_state() not in _EDITABLE_STATES:
            raise StateError("Cannot change go_back while the scan is running or paused")
        self._go_back = bool(value)

    async def _get_scan_plan(self):
        from concert.session.utils import get_default_table
        table = get_default_table(['Axis', 'Parameter', 'Start', 'Stop', 'Step', 'Points', 'Values'])
        for name, axis in self.axes.items():
            parameter = await axis.get_parameter()
            table.add_row([name, parameter.name, await axis.get_start(), await axis.get_stop(),
                           await axis.get_step(), await axis.get_num_points(),
                           await axis.get_values()])
        return f"Order: {await self.get_order()}\n" + table.get_string()

    async def _get_scan_arguments(self):
        parameters = [await axis.get_parameter() for axis in self.axes.values()]
        values = [await axis.get_values() for axis in self.axes.values()]
        return parameters, values
