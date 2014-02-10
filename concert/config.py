"""
.. data:: DISABLE_ASYNC

    Disable asynchronous execution by returning a dummy future which is not
    executed synchronusly.

.. data:: DISABLE_GEVENT

    Turn of gevent support and fall back to ThreadPoolExecutor approach.
"""

DISABLE_ASYNC = False
DISABLE_GEVENT = True
