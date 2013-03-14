def slow(fn):
    """Mark a test method as slow running.

    You can skip these test cases with nose by running ``nosetest -a '!slow'``
    or calling ``make check-fast``.
    """
    fn.slow = 1
    return fn
