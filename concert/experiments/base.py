"""
An experiment can be run multiple times. The base :py:class:`Experiment`
takes care of proper logging structure.
"""

import os
import re
import logbook
from logbook.handlers import FileHandler
from concert.helpers import async
from concert.storage import create_folder


LOGGER = logbook.Logger(__name__)


class Experiment(object):

    """
    Experiment base class. An experiment can be run multiple times
    with logging output saved on disk. The log from every
    :py:meth:`Experiment.run` can be either appended or saved
    has_multiple_foldersly, based on *root_folder* parameter.

    .. py:attribute:: run

        A callable which implements data acquisition.

    .. py:attribute:: root_folder

        Root folder of an experiment. All the data produced by this
        experiment are stored in this folder or its subfolders. If
        the parameter is formattable (see Python string) it means that
        every experiment run creates a new subfolder and log file.

    .. py:attribute:: log_file_name

        Log file name used for storing logging information.
    """

    def __init__(self, run, root_folder, log_file_name="experiment.log"):
        self.root_folder = root_folder
        self.log_file_name = log_file_name
        pattern = re.compile(".*\{.*\}.*")
        self.has_multiple_folders = pattern.match(self.root_folder) is not None
        self._folder = None
        self.file_stream = None
        self._iteration = 0
        self._run = run

    @property
    def iteration(self):
        """Get experiment iteration."""
        return self._iteration

    @property
    def folder(self):
        """Current folder for running the experiment."""
        if self.has_multiple_folders:
            folder = os.path.join(self.root_folder.format(self.iteration))
        else:
            folder = self.root_folder

        return folder

    def _create_stream_handler(self):
        """
        Create file stream handler for given scan to log information
        to the current folder.
        """
        path = os.path.join(self.folder, self.log_file_name)

        if self.has_multiple_folders and self.file_stream is not None:
            self.file_stream.close()
        if self.has_multiple_folders or self.file_stream is None:
            self.file_stream = FileHandler(path, level=logbook.INFO,
                                           bubble=True)

    @async
    def run(self, *args, **kwargs):
        """Run the experiment with logging to file."""
        # Create folder for next scan
        create_folder(self.folder)
        # Initiate new logger for this scan
        self._create_stream_handler()

        with self.file_stream:
            # All logging output goes into the file stream handler as well
            LOGGER.info("{}. experiment run".format(self.iteration + 1))
            self._run(self.folder, *args, **kwargs)
            self._iteration += 1
