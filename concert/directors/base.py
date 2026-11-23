import asyncio
import logging
from abc import abstractmethod

from concert.base import background, Parameter, StateError
from concert.runnable import ManagedRunnable

LOG = logging.getLogger(__name__)


class Director(ManagedRunnable):
    """
    Class to handle multiple experiment executions.
    """
    number_of_iterations = Parameter()
    current_iteration = Parameter()
    current_iteration_name = Parameter()

    async def __ainit__(self, experiment, walker=None, separate_runs=True,
                        name_fmt='director_run_{:>04}', **kwargs):
        """
        :param experiment: Experiment that is run. If the experiment features a
            'ready_to_prepare_next_outer_loop' event (asyncio.Event) this will be waited within the
            experiment execution. When set() the next iteration will be prepared while the
            experiment is still running. This could be used to prepare a future iteration while
            still data is stored or processed.
            The separate_runs property of the experiment should be set to False, since the director
            handles the naming of the sub-folders.
        :type experiment: concert.experiments.base.Experiment
        """
        self._experiment = experiment
        self._iteration = 0
        self._run_event = asyncio.Event()
        # Let us run by default
        self._run_event.set()
        if walker is None:
            walker = experiment.walker
        await super().__ainit__(
            walker=walker,
            separate_runs=separate_runs,
            name_fmt=name_fmt,
            **kwargs,
        )
        await self.set_log_file_prefix("director")

    async def _get_state(self):
        state = await super()._get_state()
        if state == 'running' and not self._run_event.is_set():
            return 'paused'
        else:
            return state

    @abstractmethod
    async def _prepare_run(self, iteration: int):
        """
        This function changes whatever should be different between the different experiment
        executions.

        This could trigger an exchange of the specimen or adjust parameters of the experiment.
        :param iteration:
        :type iteration: int
        """
        ...

    @abstractmethod
    async def _get_number_of_iterations(self) -> int:
        """Return the total number of iterations."""
        ...

    async def _get_current_iteration_name(self) -> str:
        return await self.get_iteration_name(await self.get_current_iteration())

    async def get_iteration_name(self, iteration: int) -> str:
        """
        Function for giving meaningfully names for each experiment execution.
        Should be overwritten for more complicated naming (e.g. specimen descriptions).
        :param iteration:
        :type  iteration: int
        :return: Name
        """
        return f"iteration_{iteration:04d}"

    async def _prepare_next_run(self):
        if await self.get_current_iteration() < await self.get_number_of_iterations() - 1:
            await self._prepare_run(await self.get_current_iteration() + 1)

    @background
    async def _run(self):
        await self._experiment['separate_runs'].stash()
        await self._experiment.set_separate_runs(False)
        try:
            self._iteration = 0
            await self._prepare_run(self._iteration)

            for iteration in range(await self.get_number_of_iterations()):
                self._iteration = iteration
                await self._run_event.wait()

                await self._experiment.walker.descend(await self.get_iteration_name(iteration))
                self._experiment.ready_to_prepare_next_outer_loop.clear()
                exp_run = self._experiment.run()
                iteration_name = await self.get_iteration_name(iteration)
                self._experiment.log.info(f"Iteration name: {iteration_name}")

                try:
                    # Without early readiness signaling, the event is set when exp_run finishes.
                    await self._experiment.ready_to_prepare_next_outer_loop.wait()
                    await self._prepare_next_run()
                    await exp_run
                except Exception as e:
                    itr_name = await self.get_iteration_name(await self.get_iteration())
                    self.log.error(
                        f"Director iteration {itr_name} failed.")
                    self.log.error(e)
                    raise e
                finally:
                    await self._experiment.walker.ascend()

        except asyncio.CancelledError:
            # This is normal, no special state needed -> standby
            LOG.warning('Experiment director cancelled')
        except Exception as e:
            # Something bad happened, and we can't know what, so set the state to error
            LOG.warning(f"Error `{e}' while running experiment director")
            raise StateError('error', msg=str(e))
        except KeyboardInterrupt:
            LOG.warning('Experiment director cancelled by keyboard interrupt')
            raise
        finally:
            await self._experiment['separate_runs'].restore()

    async def _get_current_iteration(self) -> int:
        return self._iteration

    @background
    async def pause(self):
        """
        Waits (after the current iteration is done and the next is prepared) with the next iteration
        until resume() is called.
        """
        self._run_event.clear()

    @background
    async def resume(self):
        """
        Resumes a currently paused director run.
        """
        self._run_event.set()
