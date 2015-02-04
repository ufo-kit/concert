"""
.. data:: ENABLE_ASYNC

    Enable asynchronous execution. If disabled, dummy futures are used that do
    no execute synchronously.

.. data:: ENABLE_GEVENT

    Turn on gevent support. If geven is not available, fall back to
    ThreadPoolExecutor approach.

.. data:: ENABLE_PRINT_ASYNC_EXCEPTION

    If enabled exceptions from async functions will be printed.
"""

ENABLE_ASYNC = True
ENABLE_GEVENT = False
ENABLE_PRINT_ASYNC_EXCEPTION = True
