import functools


def coroutine(func):
    """
    Start a coroutine automatically without the need to call
    next() or send(None) first.
    """
    @functools.wraps(func)
    def start(*args, **kwargs):
        """Starts the generator."""
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return start


def inject(generator, consumer):
    """
    Let a *generator* produce a value and forward it to *consumer*.
    """
    for item in generator:
        consumer.send(item)


@coroutine
def broadcast(*consumers):
    """
    broadcast(*consumers)

    Forward data to all *consumers*.
    """
    while True:
        item = yield
        for consumer in consumers:
            consumer.send(item)


class Command(object):
    """Command class for the CLI script"""

    def __init__(self, name, opts):
        """
        Command objects are loaded at run-time and injected into Concert's
        command parser.

        *name* denotes the name of the sub-command parser, e.g. "mv" for the
        MoveCommand. *opts* must be an argparse-compatible dictionary
        of command options.
        """
        self.name = name
        self.opts = opts

    def run(self, *args, **kwargs):
        """Run the command"""
        raise NotImplementedError


class Bunch(object):
    """Encapsulate a list or dictionary to provide attribute-like access.

    Common use cases look like this::

        d = {'foo': 123, 'bar': 'baz'}
        b = Bunch(d)
        print(b.foo)
        >>> 123

        l = ['foo', 'bar']
        b = Bunch(l)
        print(b.foo)
        >>> 'foo'
    """
    def __init__(self, values):
        if isinstance(values, list):
            values = dict(zip(values, values))
        self.__dict__.update(values)


def memoize(func):
    """
    Memoize the result of *func*.

    Remember the result of *func* depending on its arguments. Note, that this
    requires that the function is free from any side effects, e.g. returns the
    same value given the same arguments.
    """
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]

        result = func(*args)
        memo[args] = result
        return result

    return wrapper
