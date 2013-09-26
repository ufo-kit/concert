"""Imaging experiments usually conducted at synchrotrons."""
import time
from concert.processes.base import inject, multicast, coroutine
from concert.processes.sinks import write_images
import os
from concert.storage import write_tiff, create_folder
import numpy as np

DARKS = "darks"
FLATS = "flats"
RADIOS = "radios"


def acquire_images_dummy(num_images, images, value=None):
    """
    Acquire *num_images* and store them in *images*. If *value*
    is provided, the images will have every pixel set to *value*,
    otherwise the pixels are filled with random numbers.
    """
    for i in range(num_images):
        if value is None:
            image = np.random.random(size=(512, 512))
        else:
            image = np.ones((512, 512)) * value
        images.append(image.astype(np.float32))


def readout_images_dummy(images):
    """Readout *images* in a generator."""
    for image in images:
        yield image


def take_dummy_continuous(num_images, value=None):
    """
    Take *num_images* of dummy images filled with *value* if not None,
    otherwise random numbers.
    """
    images = []
    acquire_images_dummy(num_images, images, value=value)
    for image in readout_images_dummy(images):
        yield image


def acquire_images(camera, num_images):
    """Acquire *num_images* with a libuca *camera*."""
    try:
        camera.trigger_mode = camera.uca.props.trigger_mode.AUTO
        camera.start_recording()
        time.sleep(num_images / camera.frames_per_second)
    finally:
        camera.stop_recording()


def readout_images(camera):
    """Readout images from libuca *camera* in a generator."""
    num_frames = camera.recorded_frames.magnitude

    try:
        camera.start_readout()

        for i in xrange(num_frames):
            frame = camera.grab()

            if frame is not None:   # double check
                yield frame
    finally:
        camera.stop_readout()


def take_continuous(camera, num_images):
    """Take *num_images* continuously using a libuca *camera*."""
    acquire_images(camera, num_images)
    for image in readout_images(camera):
        yield image


def add_writer(folder, take_images, process_images, image_type,
               writer=write_tiff):
    """
    Add image writer to sinks. The resulting folder of the data
    is obtained by joining the *folder* and a subfolder determined
    by *image_type*. *process_images* is an originally assigned
    image processing coroutine. *writer* specifies which image
    writer will be used, and thus also the file type.
    """
    folder = os.path.join(folder, image_type)
    create_folder(folder)
    sinks = [write_images(writer=writer, prefix=os.path.join(folder,
                                                             image_type[:-1] +
                                                             "_{:>05}"))]
    if process_images is not None:
        sinks.append(process_images)

    inject(take_images, multicast(*sinks))


def add_averager(process_image, flat_correction):
    """
    Add a coroutine for image averaging to originally assigned
    *process_image* coroutine if *flat_correction* is True.
    """
    averager = None
    sinks = []

    if flat_correction:
        averager = ImageAverager()
        sinks.append(averager.average_images())
    if process_image is not None:
        sinks.append(process_image)

    return averager, multicast(*sinks)


def execute(folder, take_darks=None, take_flats=None, take_radios=None,
            process_darks=None, process_flats=None, process_radios=None,
            flat_correction=False, writer=write_tiff):
    """Execute one run of an imaging experiment. *folder* is the root
    folder for storing data, *take_darks* is a generator which takes
    dark field, similarly for flat fields and radiographs by *take_flats*
    and *take_radios*. *process_darks*, *process_flats*
    and *process_radios* are coroutines which define how are the
    images processed. If *flat_correction* is True, the radiographs
    are first flat corrected and then send to *process_radios*.
    *writer* specifies image writer which will handle the image storage.

    Every image is stored in a subfolder based on its type, i.e.::

        root_folder/darks
        root_folder/flats
        root_folder/radios

    folders exist after the run and are filled with images of a particular
    type. Note that radiographs are stored as they are taken without
    flat corrections even if *flat_correction* is True.
    """
    if flat_correction and (take_darks is None or take_flats is None
                            or take_radios is None):
        raise ValueError("Cannot flat correct, insufficient image sources")

    if take_darks is not None:
        dark_averager, _process_darks = add_averager(process_darks,
                                                     flat_correction)
        add_writer(folder, take_darks, _process_darks, DARKS, writer=writer)
    if take_flats is not None:
        flat_averager, _process_flats = add_averager(process_flats,
                                                     flat_correction)
        add_writer(folder, take_flats, _process_flats, FLATS, writer=writer)
    if take_radios is not None:
        if flat_correction and process_radios is not None:
            _process_radios = flat_correct(process_radios,
                                           dark_averager.average,
                                           flat_averager.average)
        else:
            _process_radios = process_radios
        add_writer(folder, take_radios, _process_radios, RADIOS, writer=writer)


class ImageAverager(object):

    """Average images in a coroutine without knowing how many will come."""

    def __init__(self):
        self.average = None

    @coroutine
    def average_images(self):
        """Average images as they come."""
        i = 0
        while True:
            data = yield
            if self.average is None:
                self.average = np.zeros_like(data, dtype=np.float32)
            self.average = (self.average * i + data) / (i + 1)
            i += 1


@coroutine
def flat_correct(sink, dark, flat):
    """
    Flat correction intermediate coroutine. It takes a *dark*, a *flat*,
    gets a radiograph from a generator, calculates flat corrected image
    and sends it forward to *sink*.
    """
    while True:
        radio = yield
        sink.send((radio - dark) / (flat - dark))
