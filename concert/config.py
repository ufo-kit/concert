"""
.. data:: ENABLE_ASYNC

    Enable casynchronous execution. If disabled, dummy futures are used that do
    no execute synchronously.

.. data:: MOTOR_VELOCITY_SAMPLING_TIME

    Time step for calculation of motor velocity by measuring two postion
    values. Longer values will create more acurate results but reading the
    velocity will take more time.

.. data:: PROGRESS_BAR

    Turn on progress bar by long-lasting operations if tqdm package is present
"""
from concert.quantities import q

ENABLE_ASYNC = True
# Prints the exception source by fake futures
PRINT_NOASYNC_EXCEPTION = True
PROGRESS_BAR = True

MOTOR_VELOCITY_SAMPLING_TIME = 0.1 * q.s
