"""
.. data:: MOTOR_VELOCITY_SAMPLING_TIME

    Time step for calculation of motor velocity by measuring two postion
    values. Longer values will create more acurate results but reading the
    velocity will take more time.

.. data:: PROGRESS_BAR

    Turn on progress bar by long-lasting operations if tqdm package is present

.. data:: GUI_EVENT_REFRESH_PERIOD

    After this time period some gui-events are processed.

.. data:: GUI_MAX_PROCESSING_TIME_FRACTION

    Maximum time fraction of GUI_EVENT_REFRESH_PERIOD that is used for the handling of the gui-events.
"""
from concert.quantities import q

# Prints the exception source by fake futures
PROGRESS_BAR = True

MOTOR_VELOCITY_SAMPLING_TIME = 0.1 * q.s

# Logging
AIODEBUG = 9
PERFDEBUG = 8

# GUI
GUI_EVENT_REFRESH_PERIOD = 0.01 * q.s
GUI_MAX_PROCESSING_TIME_FRACTION = 0.5

# Metadata files
ALWAYS_WRITE_JSON_METADATA_FILE = False

# Timeout for distributed tango servers in milliseconds
DISTRIBUTED_TANGO_TIMEOUT = 2 ** 21
