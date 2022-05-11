"""
.. data:: MOTOR_VELOCITY_SAMPLING_TIME

    Time step for calculation of motor velocity by measuring two position
    values. Longer values will create more accurate results but reading the
    velocity will take more time.

.. data:: PROGRESS_BAR

    Turn on progress bar by long-lasting operations if tqdm package is present

.. data:: DETECTOR_HORIZONTAL_DIRECTION

    Definition of the translation direction in the vertical detector direction (horizontal and
    perpendicular to the beam direction).

.. data:: BEAM_DIRECTION

    Definition of the translation direction in the beam direction.

.. data:: DETECTOR_VERTICAL_DIRECTION

    Definition of the translation in the vertical direction.

.. data:: ROTATION_AROUND_DETECTOR_ROWS

    Definition of the rotation around a detector row.

.. data:: ROTATION_AROUND_BEAM

    Definition of the rotation around the beam direction.

.. data:: ROTATION_AROUND_DETECTOR_COLUMNS

    Definition of the rotation around a detector column.

"""
from enum import Enum, auto
from concert.quantities import q

# Prints the exception source by fake futures
PROGRESS_BAR = True

MOTOR_VELOCITY_SAMPLING_TIME = 0.1 * q.s

# Logging
AIODEBUG = 9
PERFDEBUG = 8


# Coordinate systems
class Translations(Enum):
    x = auto(),
    y = auto(),
    z = auto()


class Rotations(Enum):
    pitch = auto(),
    roll = auto(),
    yaw = auto()


DETECTOR_HORIZONTAL_DIRECTION = Translations.x
BEAM_DIRECTION = Translations.y
DETECTOR_VERTICAL_DIRECTION = Translations.z

ROTATION_AROUND_DETECTOR_ROWS = Rotations.pitch
ROTATION_AROUND_BEAM = Rotations.roll
ROTATION_AROUND_DETECTOR_COLUMNS = Rotations.yaw
