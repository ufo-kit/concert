import functools
import sys
from concert.coroutines.base import run_in_loop


COMMANDS = {}


def create_command(func):
    """Create a command with from a coroutine function *func*. The command creates a coroutine from
    the coroutine function *func*, runs it and returns
    the result of the underlying coroutine. Usag in the session::

        cmd = create_command(DummyDevice.enable)
        cmd(dummy_instance)
        # You can bind it to a concrete instance of an object, then you don't need to specify
        # it as an argument (but you cannot use the command with other instances of type
        # :class:`.concert.devices.dummy.DummyDevice`
        cmd_concrete = create_command(dummy_instance.enable)
        cmd_concrete()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return run_in_loop(func(*args, **kwargs))

    return wrapper


def command(name=None):
    """Create a command with *name* from a coroutine function *func* (the argument of the nested
    `register_command` nested function which actually adds the command to the available commands
    dictionary). The command creates a coroutine from coroutine function *func*, runs it and returns
    the result of the underlying coroutine. In the session, the command can be invoked with
    name(*args, **kwargs). The original *func* is left untouched.
    """
    def register_command(func):
        nonlocal name
        if not name:
            name = func.__name__
        if name in COMMANDS:
            raise CommandError(f"Command `{name}' already registered")

        COMMANDS[name] = create_command(func)
        setattr(sys.modules[__name__], name, COMMANDS[name])

        # This magic depends on the fact that IPython injects `get_ipython` into
        # the global namespace. So, if a user imports some code into the session the comamnds will
        # be injected into the global user namespace.
        try:
            get_ipython().user_global_ns[name] = COMMANDS[name]
        except NameError:
            pass

        return func

    return register_command


class CommandError(Exception):
    pass
