========
Sessions
========

Each beamline has a certain set of motors that are used in typical experiments.
Such a group of motors can be managed with the session mechanism that is
provided by Concert and the ``concert`` command line tool. If you run the tool
without any arguments, a regular IPython shell will appear.

The following session management commands are available:

.. cmdoption:: init <name>

    Create a new session file and add initial information to it.

    *Additional options*:

    .. cmdoption:: --force

        Create the session even if one already exists with this name.

.. cmdoption:: edit <name>

    Edit the session file by launching ``$EDITOR`` with the associated Python
    module file. This file can contain any kind of Python code, but you will
    most likely just add device definitions such as this::

        from concert.devices.axes.crio import LinearAxis

        crio1 = LinearAxis(None)

.. cmdoption:: start <name>

    Load the session file and launch an IPython shell. Every definition that was
    made in the module file is available via the ``m`` variable. Moreover, the
    quantities package is already loaded and named ``q``. So, once the session
    has started you could access motors like this::

        $ concert start tomo

        This is session tomo
        Welcome to Concert 0.0.1
        In [1]: m.crio1.set_positon(2.23 * q.mm)
        In [2]: m.crio1.get_position()
        Out[2]: array(2.23) * mm

.. cmdoption:: rm <name> [<name> ...]

    Remove sessions.

    .. note::

        Be careful. The session file is unlinked from the file system and no
        backup is made.

.. cmdoption:: show

    List all available sessions.
