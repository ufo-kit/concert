"""A synchrotron beam line specific processes."""
import asyncio
import numpy as np
from concert.quantities import q
from concert.coroutines.base import background
from concert.imageprocessing import compute_rotation_axis, flat_correct
from concert.helpers import expects, Numeric
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.devices.cameras.base import Camera
from concert.devices.shutters.base import Shutter


@background
@expects(Camera, Shutter)
async def acquire_dark(camera, shutter):
    """Use *camera* and *shutter* to acquire a dark field."""
    if await shutter.get_state() != 'closed':
        await shutter.close()

    return await _grab_software_trigger(camera)


@background
@expects(Camera, Shutter, LinearMotor, Numeric(1, q.mm))
async def acquire_image_with_beam(camera, shutter, flat_motor, position):
    """Use *camera*, *shutter* and *flat_motor* to move the sample to the *position* and acquire an
    image.
    """
    coros = []
    if await shutter.get_state() != 'open':
        coros.append(shutter.open())
    coros.append(flat_motor.set_position(position))
    await asyncio.gather(*coros)

    return await _grab_software_trigger(camera)


@background
@expects(Camera, Shutter, LinearMotor, RotationMotor, Numeric(1, q.mm), Numeric(1, q.mm))
async def determine_rotation_axis(camera, shutter, flat_motor, rotation_motor, flat_position,
                                  radio_position):
    """Determine tomographic rotation axis using *camera*, *shutter*, *rotation_motor* and
    *flat_motor* devices.  *flat_position* is the position for making a flat field, *radio_position*
    is the position with the sample in the field of view.
    """
    dark = (await acquire_dark(camera, shutter)).astype(np.float32)
    flat = await acquire_image_with_beam(camera, shutter, flat_motor, flat_position)
    first = await acquire_image_with_beam(camera, shutter, flat_motor, radio_position)
    await rotation_motor.move(180 * q.deg)
    last = await acquire_image_with_beam(camera, shutter, flat_motor, radio_position)

    # flat correct
    flat_corr_first = flat_correct(first, flat, dark=dark)
    flat_corr_last = flat_correct(last, flat, dark=dark)

    return compute_rotation_axis(flat_corr_first, flat_corr_last)


async def _grab_software_trigger(camera):
    """Switch *camera* to software trigger source and take an image. The source is restored afterwards.
    """
    await camera['trigger_source'].stash()
    await camera.set_trigger_source(camera.trigger_sources.SOFTWARE)

    try:
        await camera.trigger()
        return await camera.grab()
    finally:
        await camera['trigger_source'].restore()
