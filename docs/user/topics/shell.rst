==================
Command line shell
==================

Concert comes with a command line interface that is launched by typing
``concert`` into a shell. Several subcommands define the action of the tool.


Session commands
================

.. program:: concert

.. option:: init <session>

Create a new session.

    .. option:: --force

        Create the session even if one already exists with this name.

    .. option:: --imports

        List of module names that are added to the import list.


.. option:: edit <session>

Edit the session file by launching ``$EDITOR`` with the associated Python
module file. This file can contain any kind of Python code, but you will
most likely just add device definitions such as this::

    from concert.devices.axes.crio import LinearAxis

    crio1 = LinearAxis(None)


.. option:: log <session>

Show session logs. If a *session* is not given, the log command shows entries
from all sessions.


.. option:: show <session> <session>

Show all available sessions or details of a given *session*.


.. option:: mv <source session> <target session>

Move session *source* to *target*.


.. option:: rm <session> <session>

Remove one or more sessions.  

.. warning::

    Be careful. The session file is unlinked from the file system and no
    backup is made.


.. option:: fetch <path or url>

Import an existing *session*.

    .. option:: --force

        Overwrite session if it already exists.

    .. option:: --repo

        The URL denotes a Git repository from which the sessions are imported.


.. option:: start <session>

Load the session file and launch an IPython shell.  The quantities package is
already loaded and named ``q``.

    .. option:: --logto={stderr, file}

        Specify a method for logging events. If this flag is not specified,
        ``file`` is used and assumed to be
        ``$XDG_DATA_HOME/concert/concert.log``.

    .. option:: --logfile=<filename>

        Specify a log file if ``--logto`` is set to ``file``.

    .. option:: --loglevel={debug, info, warning, error, critical}

        Specify lowest log level that is logged.

    .. cmdoption:: --non-interactive

        Run the session as a script and do not launch a shell.


Extensions
==========

Spyder
------

.. option:: spyder <session>

If Spyder is installed, start the *session* within the Spyder GUI.
