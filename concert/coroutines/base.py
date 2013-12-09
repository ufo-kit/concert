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
