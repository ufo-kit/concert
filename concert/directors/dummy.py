import logging
from concert.directors.base import Director as BaseDirector

LOG = logging.getLogger(__name__)


class Director(BaseDirector):
    """
    Dummy director, that runs an Experiment *num_iteration* times.
    """
    async def __ainit__(self, experiment, num_iterations: int):
        """
        :param experiment: Experiment instance
        :param num_iterations: Number of experiment runs.
        """
        self._num_iterations = int(num_iterations)
        await super().__ainit__(experiment)

    async def _get_number_of_iterations(self) -> int:
        return self._num_iterations

    async def _prepare_run(self, iteration: int):
        self.log.info(f"Preparing iteration {iteration}.")
