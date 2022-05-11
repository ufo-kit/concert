import os
import asyncio
import logging

from concert.base import Parameterizable, background, Parameter, State, transition, StateError, \
    check, Selection

LOG = logging.getLogger(__name__)


class Director(Parameterizable):
    """
    Class to handle multiple experiment executions.
    """
    state = State(default="standby")
    number_of_iterations = Parameter()
    current_iteration = Parameter()
    log_level = Selection(['critical', 'error', 'warning', 'info', 'debug'])

    async def __ainit__(self, experiment):
        """
        :param experiment: Experiment that is run. If the experiment features a
            'ready_to_prepare_next_sample' event (asyncio.Event) this will be waited within the
            experiment execution. When set() the next iteration will be prepared while the
            experiment is still running. This could be used to prepare a future iteration while
            still data is stored or processed.
            The separate_scans property of the experiment should be set to False, since the director
            handles the naming of the sub-folders.
        :type experiment: concert.experiments.base.Experiment
        """
        self._experiment = experiment
        self._run_event = asyncio.Event()
        # Let us run by default
        self._run_event.set()
        self._iteration = 0
        self.log = LOG
        self.log.setLevel("INFO")
        await super().__ainit__()

    async def _get_log_level(self):
        return logging.getLevelName(self.log.getEffectiveLevel()).lower()

    async def _set_log_level(self, level):
        self.log.setLevel(level.upper())

    async def _prepare_run(self, iteration: int):
        """
        This function changes whatever should be different between the different experiment
        executions.

        This could trigger an exchange of the specimen or adjust parameters of the experiment.
        :param iteration:
        :type iteration: int
        """
        raise NotImplementedError

    async def _get_number_of_iterations(self) -> int:
        """
        Should return the total number of iterations.
        :return: Number of iterations.
        """
        raise NotImplementedError

    async def _get_iteration_name(self, iteration: int) -> str:
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
    @check(source=['standby', 'error'], target="standby")
    @transition(immediate="running", target="standby")
    async def run(self):
        handler = None
        try:
            if self._experiment.walker:
                handler = logging.FileHandler(os.path.join(self._experiment.walker.current,
                                                           'director.log'))
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                              '- %(message)s')
                handler.setFormatter(formatter)
                self.log.addHandler(handler)
            self.log.info(await self.info_table)

            await self.prepare()
            # prepare first iteration
            self._iteration = 0
            await self._prepare_run(self._iteration)

            for iteration in range(await self.get_number_of_iterations()):
                self._iteration = iteration
                if not self._run_event.is_set():
                    self._state_value = "paused"
                await self._run_event.wait()
                self._state_value = "running"

                self._experiment.walker.descend(await self._get_iteration_name(iteration))
                exp_run = self._experiment.run()
                sample_name = await self._get_iteration_name(iteration)
                self._experiment.log.info(f"Sample name: {sample_name}")
                self._experiment.log.info(await self.info_table)

                # Note: exp_run and self._experiment.ready_to_prepare_next_sample.wait() finish
                # at the same time, if the user does not implement ready_to_prepare_next_sample.
                await self._experiment.ready_to_prepare_next_sample.wait()
                await self._prepare_next_run()
                await exp_run

        except asyncio.CancelledError:
            # This is normal, no special state needed -> standby
            LOG.warning('Experiment director cancelled')
            self._state_value = 'standby'
        except Exception as e:
            # Something bad happened, and we can't know what, so set the state to error
            LOG.warning(f"Error `{e}' while running experiment director")
            raise StateError('error', msg=str(e))
        except KeyboardInterrupt:
            LOG.warning('Experiment director cancelled by keyboard interrupt')
            self._state_value = 'standby'
            raise
        finally:
            if handler:
                handler.close()
                self.log.removeHandler(handler)
            self._experiment.walker.ascend()
            await self.finish()

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

    async def prepare(self):
        pass

    async def finish(self):
        pass
