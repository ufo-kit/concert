"""
.. data:: ENABLE_ASYNC

    Enable asynchronous execution. If disabled, dummy futures are used that do
    no execute synchronously.

.. data:: ENABLE_GEVENT

    Turn on gevent support. If geven is not available, fall back to
    ThreadPoolExecutor approach.

.. data:: PROGRESS_BAR

    Turn on progress bar by long-lasting operations if tqdm package is present
"""

ENABLE_ASYNC = True
ENABLE_GEVENT = False
# Prints the exception source by fake futures
PRINT_NOASYNC_EXCEPTION = True
PROGRESS_BAR = True
