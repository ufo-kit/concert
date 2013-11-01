"""
An experiment can be run multiple times. The base :py:class:`Experiment`
takes care of proper logging structure.
"""

import os
import re
import logging
from logging import FileHandler, Formatter
from concert.storage import create_directory


LOG = logging.getLogger(__name__)


class Experiment(object):

    """
    Experiment base class. An experiment can be run multiple times
    with logging output saved on disk. The log from every
    :py:meth:`Experiment.run` can be either appended or saved
    has_multiple_directories, based on *root_directory* parameter.

    .. py:attribute:: run

        A callable which implements data acquisition.

    .. py:attribute:: root_directory

        Root directory of an experiment. All the data produced by this
        experiment are stored in this directory or its subdirectories. If
        the parameter is formattable (see Python string) it means that
        every experiment run creates a new subdirectory and log file.

    .. py:attribute:: log_file_name

        Log file name used for storing logging information.

    .. py:attribute:: iteration

        Iteration number to start with. If the experiment runs scans in
        separate directories then the first scan directory index will be
        the given number.

    """

    def __init__(self, run, root_directory, iteration=1,
                 log_file_name="experiment.log"):
        self.root_directory = root_directory
        self.log_file_name = log_file_name
        pattern = re.compile(".*\{.*\}.*")
        self.has_multiple_directories = \
            pattern.match(self.root_directory) is not None
        self._directory = None
        self.file_stream = None
        self.iteration = iteration
        self._run = run

    @property
    def directory(self):
        """Current directory for running the experiment."""
        if self.has_multiple_directories:
            directory = os.path.join(
                self.root_directory.format(self.iteration))
        else:
            directory = self.root_directory

        return directory

    def _create_stream_handler(self):
        """
        Create file stream handler for given scan to log information
        to the current directory.
        """
        path = os.path.join(self.directory, self.log_file_name)
        root_logger = logging.getLogger("")

        if self.has_multiple_directories and self.file_stream is not None:
            self.file_stream.close()
            root_logger.removeHandler(self.file_stream)
        if self.has_multiple_directories or self.file_stream is None:
            self.file_stream = FileHandler(path)
            self.file_stream.setLevel(logging.INFO)
            formatter = Formatter("[%(asctime)s] %(levelname)s: " +
                                  "%(name)s: %(message)s")
            self.file_stream.setFormatter(formatter)
            root_logger.addHandler(self.file_stream)

    def run(self, *args, **kwargs):
        """
        Run the experiment with logging to file, *args* and *kwargs* are
        arguments and keyword arguments to be passed to the method which
        actually conducts the experiment. The method is specified in the
        constructor.
        """
        # Create directory for next scan
        create_directory(self.directory)
        if self.has_multiple_directories and os.listdir(self.directory):
            raise ValueError("Folder {} is not empty".format(self.directory))

        # Initiate new logger for this scan
        self._create_stream_handler()

        LOG.info("{}. experiment run".format(self.iteration))
        self._run(*args, **kwargs)
        self.iteration += 1
