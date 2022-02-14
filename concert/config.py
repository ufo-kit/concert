"""
.. data:: MOTOR_VELOCITY_SAMPLING_TIME

    Time step for calculation of motor velocity by measuring two position
    values. Longer values will create more accurate results but reading the
    velocity will take more time.

.. data:: PROGRESS_BAR

    Turn on progress bar by long-lasting operations if tqdm package is present
"""
from concert.quantities import q

# Prints the exception source by fake futures
PROGRESS_BAR = True

MOTOR_VELOCITY_SAMPLING_TIME = 0.1 * q.s

# Logging
AIODEBUG = 9
PERFDEBUG = 8
