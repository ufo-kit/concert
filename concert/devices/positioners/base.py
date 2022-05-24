"""
Base class for complex motion realized by combination of more motors
from :mod:`concert.motors.motors.base`.

The :class:`.Positioner` class facilitates multiple :class:`.Axis` for
translation and rotation which form the motion possibilities
of a particular positioner. The coordinate system of the axes within
a positioner is the following:

    * x - horizontal coordinate in the image plane
    * y - vertical coordinate in the image plane
    * z - beam direction

The direction of movements is as follows:

    * + - right/up/forward in x/y/z
    * - - left/down/back in x/y/z

*Note*: *forward* in 'z' directions means away from the detector, one can
interpret it as by "increasing the propagation distance".

Axes which represent angular motion are also represented by a coordinate,
when "y" angular axis means we can rotate around "y" axis. We consider
the system to be "left handed", i.e. that for example the "y" axis
positive rotation is in the clockwise direction.

An exmaple of a positioner which is capable of translation in all
coordinates and rotation around "y"::

    import numpy as np
    from concert.devices.motors.dummy import LinearMotor, RotationMotor

    x_axis = Axis('x', LinearMotor())
    y_axis = Axis('y', LinearMotor())
    z_axis = Axis('z', LinearMotor())
    y_rot_axis = Axis('y', RotationMotor())
    positioner = Positioner([x_axis, y_axis, z_axis, y_rot_axis])

    # Move to the center of origin
    positioner.position = (0, 0, 0) * q.mm
    # Move to the right
    positioner.position = (10, 0, 0) * q.mm
    # Rotate 90 degrees counter clockwise
    positioner.orientation = (np.nan, - 90, np.nan) * q.deg

"""
import asyncio
import numpy as np
from concert.coroutines.base import background
from concert.quantities import q
from concert.base import AsyncObject, Quantity
from concert.devices.base import Device
from concert.devices.motors.base import LinearMotor


INF_VECTOR = np.array((np.inf, np.inf, np.inf))


class Axis(AsyncObject):
    """
    An axis represents a Euclidean axis along which one can translate or
    around which one can rotate. The axis *coordinate* is a string representing
    the Euclidean axis, i.e. 'x' or 'y' or 'z'. Movement is realized by a *motor*.
    An additional *position* argument is necessary for calculatin more complicated
    motion types, e.g. rotation around arbitrary point in space. It is the local
    position with respect to a :class:`concert.devices.positioners.base.Positioner`
    in which it is placed.
    """
    async def __ainit__(self, coordinate, motor, direction=1, position=None):
        self.coordinate = coordinate
        self.motor = motor
        self.direction = direction
        self.position = position

    @background
    async def get_position(self):
        """
        get_position()

        Get position asynchronously with respect to axis direction.
        """
        return (await self.motor.get_position()) * self.direction

    @background
    async def set_position(self, position):
        """Set the *position* asynchronously with respect to axis direction."""
        return await self.motor.set_position(position * self.direction)


class Positioner(Device):
    """Combines more motors which move to form a complex motion. *axes* is a list
    of :class:`.Axis` instances. *position* is a 3D vector of coordinates specifying
    the global position of the positioner.

    If a certain coordinate in the positioner is missing, then when we set the
    position or orientation we can specify the respective vector position to
    be zero or numpy.nan.
    """

    position = Quantity(q.m, help="Global position",
                        lower=-INF_VECTOR * q.m, upper=INF_VECTOR * q.m)

    orientation = Quantity(q.rad, help="Orientation of the coordinate system",
                           lower=-INF_VECTOR * q.rad, upper=INF_VECTOR * q.rad)

    async def __ainit__(self, axes, position=None):
        await super(Positioner, self).__ainit__()
        self.translators = {}
        self.rotators = {}
        self.global_position = None

        for axis in axes:
            if isinstance(axis.motor, LinearMotor):
                self.translators[axis.coordinate] = axis
            else:
                self.rotators[axis.coordinate] = axis

    @background
    async def move(self, position):
        """
        move(position)

        Move by specified *position*.
        """
        await self.set_position(await self.get_position() + position)

    @background
    async def rotate(self, angles):
        """
        rotate(angles)

        Rotate by *angles*.
        """
        await self.set_orientation(await self.get_orientation() + angles)

    @background
    async def right(self, value):
        """
        right(value)

        Move right by *value*."""
        return await self.move(_vectorize(value, 'x'))

    @background
    async def left(self, value):
        """
        left(value)

        Move left by *value*."""
        return await self.right(-value)

    @background
    async def up(self, value):
        """
        up(value)

        Move up by *value*.
        """
        return await self.move(_vectorize(value, 'y'))

    @background
    async def down(self, value):
        """
        down(value)

        Move down by *value*.
        """
        return await self.up(-value)

    @background
    async def forward(self, value):
        """
        forward(value)

        Move forward by *value*.
        """
        return await self.move(_vectorize(value, 'z'))

    @background
    async def back(self, value):
        """
        back(value)

        Move back by *value*.
        """
        return await self.forward(-value)

    async def _get_position(self):
        """Get the position of the positioner."""
        return await self._get_vector(self.translators)

    async def _set_position(self, position):
        """3D translation to *position*."""
        await self._set_vector(position, self.translators)

    async def _get_orientation(self):
        """Get the angular position of the positioner."""
        return await self._get_vector(self.rotators)

    async def _set_orientation(self, angles):
        """Rotation with magnitudes *angles*."""
        await self._set_vector(angles, self.rotators)

    async def _get_vector(self, axes):
        """Get the current translation or orientation vector."""
        vector = []
        unit = q.m if axes == self.translators else q.rad

        for coordinate in ['x', 'y', 'z']:
            if coordinate in axes:
                vector.append((await axes[coordinate].get_position()).to(unit).magnitude)
            else:
                vector.append(np.nan)

        return vector * unit

    async def _set_vector(self, vector, axes):
        """Set position (angular or translational) given by *vector* on *axes*."""
        coros = []
        for i, coordinate in enumerate(['x', 'y', 'z']):
            magnitude = vector[i].magnitude
            if not np.isnan(magnitude):
                if coordinate not in axes:
                    if magnitude != 0:
                        # Last chance is to specify the coordinate to be zero.
                        raise PositionerError('Cannot move in {} coordinate'.format(coordinate))
                else:
                    coro = axes[coordinate].set_position(vector[i])
                    coros.append(coro)

        await asyncio.gather(*coros)


class PositionerError(Exception):
    """Positioning exceptions"""
    pass


def _vectorize(scalar, coordinate):
    """
    Return a vector with the *scalar* in the correct place given by the
    *coordinate* as x, y or z.
    """
    vector = np.zeros(3, dtype=float)
    translate = dict(x=0, y=1, z=2)
    vector[translate[coordinate]] = scalar.magnitude

    return q.Quantity(vector, scalar.units)
