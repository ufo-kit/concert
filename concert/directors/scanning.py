import asyncio
from concert.quantities import q
from concert.helpers import arange
from concert.base import Quantity, Parameter, check
from concert.directors.base import Director


class XYScan(Director):
    """
    Director to scan a specimen within a plane.
    """
    x_min = Quantity(q.mm, check=check(source=['standby', 'error']))
    x_max = Quantity(q.mm, check=check(source=['standby', 'error']))
    x_step = Quantity(q.mm, check=check(source=['standby', 'error']))
    y_min = Quantity(q.mm, check=check(source=['standby', 'error']))
    y_max = Quantity(q.mm, check=check(source=['standby', 'error']))
    y_step = Quantity(q.mm, check=check(source=['standby', 'error']))
    x_num = Parameter()
    y_num = Parameter()

    async def __ainit__(self, experiment, x_motor, y_motor, x_min, x_max, x_step,
                        y_min, y_max, y_step):
        """
        :param experiment: Experiment that is run. If the experiment features a
            'ready_to_prepare_next_sample' event (asyncio.Event) this will be waited within the
            experiment execution. When set() the next iteration will be prepared while the
            experiment is still running. This could be used to prepare a future iteration while
            still data is stored or processed.
            The separate_scans property of the experiment should be set to False, since the director
            handles the naming of the sub-folders.
        :type experiment: concert.experiments.base.Experiment
        :param x_motor: Linear motor for scanning in x direction
        :type x_motor: concert.devices.motors.base.LinearMotor
        :param y_motor: Linear motor for scanning in y direction
        :type y_motor: concert.devices.motors.base.LinearMotor
        :param x_min: Starting position of x
        :type x_min: q.mm
        :param x_max: Stop position of x
        :type x_max: q.mm
        :param x_step: Step width of x scanning
        :type x_step: q.mm
        :param y_min: Starting position of y
        :type y_min: q.mm
        :param y_max: Stop position of y
        :type y_max: q.mm
        :param y_step: Step width of y scanning
        :type y_step: q.mm
        """
        self._x_min = None
        self._x_max = None
        self._x_step = None
        self._y_min = None
        self._y_max = None
        self._y_step = None
        await super().__ainit__(experiment)
        self._x_motor = x_motor
        self._y_motor = y_motor
        await self.set_x_min(x_min)
        await self.set_x_max(x_max)
        await self.set_x_step(x_step)
        await self.set_y_min(y_min)
        await self.set_y_max(y_max)
        await self.set_y_step(y_step)

    async def _get_x_min(self):
        return self._x_min

    async def _get_x_max(self):
        return self._x_max

    async def _get_x_step(self):
        return self._x_step

    async def _get_y_min(self):
        return self._y_min

    async def _get_y_max(self):
        return self._y_max

    async def _get_y_step(self):
        return self._y_step

    async def _set_x_min(self, pos):
        self._x_min = pos

    async def _set_x_max(self, pos):
        self._x_max = pos

    async def _set_x_step(self, pos):
        self._x_step = pos

    async def _set_y_min(self, pos):
        self._y_min = pos

    async def _set_y_max(self, pos):
        self._y_max = pos

    async def _set_y_step(self, pos):
        self._y_step = pos

    async def _get_number_of_iterations(self) -> int:
        return await self.get_x_num() * await self.get_y_num()

    async def _get_x_num(self) -> int:
        return len(arange(await self.get_x_min(), await self.get_x_max(), await self.get_x_step()))

    async def _get_y_num(self) -> int:
        return len(arange(await self.get_y_min(), await self.get_y_max(), await self.get_y_step()))

    async def _prepare_run(self, iteration: int):
        x_pos = arange(await self.get_x_min(), await self.get_x_max(), await self.get_x_step())
        y_pos = arange(await self.get_y_min(), await self.get_y_max(), await self.get_y_step())
        x_index = iteration // await self.get_y_num()
        y_index = iteration % await self.get_y_num()
        await asyncio.gather(self._x_motor.set_position(x_pos[x_index]),
                             self._y_motor.set_position(y_pos[y_index]))

    async def _get_iteration_name(self, iteration: int) -> str:
        x_index = iteration // await self.get_y_num()
        y_index = iteration % await self.get_y_num()
        return f"iteration_{x_index:04d}_{y_index:04d}"
