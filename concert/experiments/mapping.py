import os.path

from concert.experiments.base import Experiment as BaseExperiment, Acquisition, _runnable_state
from concert.quantities import q
from concert.base import Parameter, Quantity, check, Parameterizable
from concert.helpers import arange
import json
import numpy as np


class Mapping(BaseExperiment):
    num_darks = Parameter(check=check(source=_runnable_state))

    async def __ainit__(self, walker, camera, separate_scans=False):
        darks_acquisition = await Acquisition('darks', self._take_darks)
        mapping_acquisition = await Acquisition('mapping', self._take_mapping)
        self._camera = camera
        self._num_darks = None
        await super().__ainit__(acquisitions=[darks_acquisition, mapping_acquisition],
                                walker=walker,
                                separate_scans=separate_scans)
        await self.set_num_darks(10)

    async def _set_num_darks(self, n: int):
        self._num_darks = int(n)

    async def _get_num_darks(self) -> int:
        return self._num_darks

    async def sample_positions(self) -> [(q.Quantity, q.Quantity)]:
        raise NotImplementedError

    async def write_positions(self, path=None):
        if path is None:
            path = self.walker.current
        positions = await self.sample_positions()
        position_dict = {}
        for i, (x, y) in enumerate(positions):
            position_dict[f"position_{i}"] = {'x': x, 'y': y}
        with open(os.path.join(path, "positions.json"), "w") as f:
            json.dump(position_dict, f)

    async def position_sample(self, position_x, position_y):
        raise NotImplementedError

    async def _take_darks(self):
        """
        Generator for taking dark images

        First :py:meth:`._prepare_darks()` is called. Afterwards :py:meth:`._produce_frames()`
        generates the frames.
        At the end :py:meth:`._finish_darks()` is called.
        """
        try:
            await self._prepare_darks()
            if await self._camera.get_state() == "recording":
                await self._camera.stop_recording()
            await self._camera.set_trigger_source("AUTO")
            async with self._camera.recording():
                for _ in range(await self.get_num_darks()):
                    yield await self._camera.grab()
        finally:
            await self._finish_darks()

    async def _prepare_darks(self):
        """
        Called before the dark images are acquired.

        Calls :py:meth:`.stop_sample_exposure()`.
        """
        await self.stop_sample_exposure()

    async def _finish_darks(self):
        """
        Called after all dark images are acquired.

        Does nothing in this class.
        """
        pass

    async def _take_mapping(self):
        positions = await self.sample_positions()
        await self.start_sample_exposure()

        if await self._camera.get_state() == "recording":
            await self._camera.stop_recording()
        await self._camera.set_trigger_source("SOFTWARE")

        async with self._camera.recording():
            for x, y in positions:
                await self.position_sample(x, y)
                await self._camera.trigger()
                yield await self._camera.grab()

        await self.stop_sample_exposure()

    async def start_sample_exposure(self):
        """
        This function must implement in a way that the sample is exposed by radiation, like opening
        a shutter or starting an X-ray tube.
        """
        raise NotImplementedError

    async def stop_sample_exposure(self):
        """
        This function must implement in a way that the sample is not exposed by radiation, like
        closing a shutter or switching off an X-ray tube.
        """
        raise NotImplementedError


class MappingPositionMixin(Parameterizable):
    effective_pixel_size = Quantity(q.um)
    field_of_view_x = Quantity(q.um)
    field_of_view_y = Quantity(q.um)
    overlap = Parameter()

    async def __ainit__(self, effective_pixel_size, field_of_view_x, field_of_view_y, overlap=0.1):

        self._effective_pixel_size = None
        self._field_of_view_x = None
        self._field_of_view_y = None
        self._overlap = None

        await Parameterizable.__ainit__(self)
        await self.set_effective_pixel_size(effective_pixel_size)
        await self.set_field_of_view_x(field_of_view_x)
        await self.set_field_of_view_y(field_of_view_y)
        await self.set_overlap(overlap)

    async def _set_effective_pixel_size(self, effective_pixel_size):
        self._effective_pixel_size = effective_pixel_size

    async def _get_effective_pixel_size(self):
        return self._effective_pixel_size

    async def _set_overlap(self, overlap: float):
        self._overlap = float(overlap)

    async def _get_overlap(self):
        return self._overlap

    async def _set_field_of_view_x(self, field_of_view_x):
        self._field_of_view_x = field_of_view_x

    async def _get_field_of_view_x(self):
        return self._field_of_view_x

    async def _set_field_of_view_y(self, field_of_view_y):
        self._field_of_view_y = field_of_view_y

    async def _get_field_of_view_y(self):
        return self._field_of_view_y


class RectangleMappingMixin(MappingPositionMixin):
    size_x = Quantity(q.um)
    size_y = Quantity(q.um)
    center_x = Quantity(q.um)
    center_y = Quantity(q.um)

    async def __ainit__(self, effective_pixel_size, size_x, size_y, center_x, center_y, overlap):
        self._size_x = None
        self._size_y = None
        self._center_x = None
        self._center_y = None

        await super().__ainit__(effective_pixel_size, size_x, size_y, overlap)

        await self.set_size_x(size_x)
        await self.set_size_y(size_y)
        await self.set_center_x(center_x)
        await self.set_center_y(center_y)

    async def _set_size_x(self, size_x):
        self._size_x = size_x

    async def _get_size_x(self):
        return self._size_x

    async def _set_size_y(self, size_y):

        self._size_y = size_y

    async def _get_size_y(self):
        return self._size_y

    async def _set_center_x(self, center_x):
        self._center_x = center_x

    async def _get_center_x(self):
        return self._center_x

    async def _set_center_y(self, center_y):
        self._center_y = center_y

    async def _get_center_y(self):
        return self._center_y

    async def sample_positions(self) -> [(q.Quantity, q.Quantity)]:
        """
        Returns a list of tuples of x and y positions in micrometer.
        x runs from center_x - size_x/2 - field_of_view_x/2 to center_x + size_x/2 + field_of_view_x/2
        y runs from center_y - size_y/2 - field_of_view_y/2 to center_y + size_y/2 + field_of_view_y/2
        step size is field_of_view * (1-overlap)
        """
        x = arange(
            start=await self.get_center_x() - await self.get_size_x() / 2 - await self.get_field_of_view_x() / 2,
            stop=await self.get_center_x() + await self.get_size_x() / 2 + await self.get_field_of_view_x() / 2,
            step=await self.get_field_of_view_x() * (1 - await self.get_overlap()))
        y = arange(
            start=await self.get_center_y() - await self.get_size_y() / 2 - await self.get_field_of_view_y() / 2,
            stop=await self.get_center_y() + await self.get_size_y() / 2 + await self.get_field_of_view_y() / 2,
            step=await self.get_field_of_view_y() * (1 - await self.get_overlap()))
        positions = []
        for x_pos in x:
            for y_pos in y:
                positions.append((x_pos, y_pos))
        return positions


class CircularMappingMixin(MappingPositionMixin):
    radius = Quantity(q.um)
    center_x = Quantity(q.um)
    center_y = Quantity(q.um)

    async def __ainit__(self, effective_pixel_size, field_of_view_x, field_of_view_y, center_x,
                        center_y, radius, overlap=0.1):
        self._center_x = None
        self._center_y = None
        self._radius = None
        await super().__ainit__(effective_pixel_size, field_of_view_x, field_of_view_y, overlap)
        await self.set_radius(radius)
        await self.set_center_x(center_x)
        await self.set_center_y(center_y)

    async def sample_positions(self) -> [(q.Quantity, q.Quantity)]:
        x = arange(
            start=await self.get_center_x() - await self.get_radius() - await self.get_field_of_view_x() / 2,
            stop=await self.get_center_x() + await self.get_radius() + await self.get_field_of_view_x() / 2,
            step=await self.get_field_of_view_x() * (1 - await self.get_overlap()))
        y = arange(
            start=await self.get_center_y() - await self.get_radius() - await self.get_field_of_view_y() / 2,
            stop=await self.get_center_y() + await self.get_radius() + await self.get_field_of_view_y() / 2,
            step=await self.get_field_of_view_y() * (1 - await self.get_overlap()))

        fov_diagonal = np.sqrt(
            await self.get_field_of_view_x() ** 2 + await self.get_field_of_view_y() ** 2)
        positions = []
        for x_pos in x:
            for y_pos in y:
                if (x_pos - await self.get_center_x()) ** 2 + (
                        y_pos - await self.get_center_y()) ** 2 <= await (
                        self.get_radius() + fov_diagonal) ** 2:
                    positions.append((x_pos, y_pos))

        return positions

    async def _set_center_x(self, center_x):
        self._center_x = center_x

    async def _get_center_x(self):
        return self._center_x

    async def _set_center_y(self, center_y):
        self._center_y = center_y

    async def _get_center_y(self):
        return self._center_y

    async def _set_radius(self, radius):
        self._radius = radius

    async def _get_radius(self):
        return self._radius


class Mapping2DMotorsMixin:
    async def __ainit__(self, x_motor, y_motor):
        self._x_motor = x_motor
        self._y_motor = y_motor

    async def set_position(self, x, y):
        await self._x_motor.set_position(x)
        await self._y_motor.set_position(y)


class RectangularMotorMapping(Mapping2DMotorsMixin, RectangleMappingMixin, Mapping):
    async def __ainit__(self, walker, camera, x_motor, y_motor, effective_pixel_size,
                        field_of_view_x, field_of_view_y, center_x,
                        center_y, size_x, size_y, overlap=0.1, separate_scans=False):
        await Mapping.__ainit__(self=self, walker=walker,
                                camera=camera, separate_scans=separate_scans)
        await Mapping2DMotorsMixin.__ainit__(self, x_motor, y_motor)
        await RectangleMappingMixin.__ainit__(self, effective_pixel_size=effective_pixel_size,
                                              center_x=center_x,
                                              center_y=center_y, size_x=size_x, size_y=size_y,
                                              overlap=overlap)
