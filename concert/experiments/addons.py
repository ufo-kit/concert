"""Add-ons for acquisitions are standalone extensions which can be applied to them. They operate on
the acquired data, e.g. write images to disk, do tomographic reconstruction etc.
"""
import os
import logging
import numpy as np
from concert.helpers import ImageWithMetadata

from concert.base import AsyncObject
from concert.coroutines.base import async_generate
from concert.coroutines.sinks import Accumulate
from concert.experiments.imaging import GratingInterferometryStepping

LOG = logging.getLogger(__name__)


class Addon(object):

    """A base addon class. An addon can be attached, i.e. its functionality is applied to the
    specified *acquisitions* and detached.

    .. py:attribute:: acquisitions

    A list of :class:`~concert.experiments.base.Acquisition` objects. The addon attaches itself on
    construction.

    """

    def __init__(self, acquisitions):
        self.acquisitions = acquisitions
        self._attached = False
        self.attach()

    def attach(self):
        """Attach the addon to all acquisitions."""
        if self._attached:
            LOG.debug('Cannot attach an already attached Addon')
        else:
            self._attach()
            self._attached = True

    def detach(self):
        """Detach the addon from all acquisitions."""
        if self._attached:
            self._detach()
            self._attached = False
        else:
            LOG.debug('Cannot detach an unattached Addon')

    def _attach(self):
        """Attach implementation."""
        raise NotImplementedError

    def _detach(self):
        """Detach implementation."""
        raise NotImplementedError


class Consumer(Addon):

    """An addon which applies a specific coroutine-based consumer to acquisitions.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: consumer

    A callable which returns a coroutine which processes the incoming data from acquisitions

    """

    def __init__(self, acquisitions, consumer):
        self.consumer = consumer
        super(Consumer, self).__init__(acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.append(self.consumer)

    def _detach(self):
        """Detach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self.consumer)


class Accumulator(Addon):

    """An addon which accumulates data.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: shapes

    a list of shapes for different acquisitions

    .. py:attribute:: dtype

    the numpy data type
    """

    def __init__(self, acquisitions, shapes=None, dtype=None):
        self._accumulators = {}
        self._shapes = shapes
        self._dtype = dtype
        self.items = {}
        super(Accumulator, self).__init__(acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        shapes = (None,) * len(self.acquisitions) if self._shapes is None else self._shapes

        for i, acq in enumerate(self.acquisitions):
            self._accumulators[acq] = Accumulate(shape=shapes[i], dtype=self._dtype)
            self.items[acq] = self._accumulators[acq].items
            acq.consumers.append(self._accumulators[acq])

    def _detach(self):
        """Detach all acquisitions."""
        self.items = {}
        for acq in self.acquisitions:
            acq.consumers.remove(self._accumulators[acq])

        self._accumulators = {}


class ImageWriter(Addon):

    """An addon which writes images to disk.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: walker

    A :class:`~concert.storage.Walker` instance
    """

    def __init__(self, acquisitions, walker):
        self.walker = walker
        self._writers = {}
        super(ImageWriter, self).__init__(acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        for acq in self.acquisitions:
            self._writers[acq] = self._write_sequence(acq)
            acq.consumers.append(self._writers[acq])

    def _detach(self):
        """Detach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self._writers[acq])
            del self._writers[acq]

    def _write_sequence(self, acquisition):
        """Wrap the walker and write data."""
        async def wrapped_writer(producer):
            """Returned wrapper."""
            async with self.walker:
                writer = self.walker.create_writer(producer, name=acquisition.name)
            await writer

        return wrapped_writer


class OnlineReconstruction(AsyncObject, Addon):
    async def __ainit__(self, experiment, reco_args, do_normalization=True,
                        average_normalization=True, walker=None, slice_directory='online-slices'):
        from concert.ext.ufo import GeneralBackprojectManager

        self.experiment = experiment
        self.manager = await GeneralBackprojectManager(
            reco_args,
            average_normalization=average_normalization
        )
        self.walker = walker
        self.slice_directory = slice_directory
        self._consumers = {}
        self._do_normalization = do_normalization
        super().__init__(experiment.acquisitions)

    async def _reconstruct(self, producer):
        await self.manager.backproject(producer)
        if self.walker:
            async with self.walker:
                producer = async_generate(self.manager.volume)
                writer = self.walker.create_writer(
                    producer,
                    name=self.slice_directory,
                    dsetname='slice_{:>04}.tif'
                )
            await writer

    def _attach(self):
        if self._do_normalization:
            self._consumers[self.experiment.darks] = self.manager.update_darks
            self._consumers[self.experiment.flats] = self.manager.update_flats
        self._consumers[self.experiment.radios] = self._reconstruct

        for acq, consumer in self._consumers.items():
            acq.consumers.append(consumer)

    def _detach(self):
        for acq, consumer in list(self._consumers.items()):
            acq.consumers.remove(consumer)


class PhaseGratingSteppingFourierProcessing(Addon):
    """
    Addon for concert.experiments.imaging.GratingInterferometryStepping to process the raw data.
    The order of the acquisitions can be changed.
    """

    def __init__(self, experiment, output_directory="contrasts"):
        if not isinstance(experiment, GratingInterferometryStepping):
            raise Exception("This addon can only be used with "
                            "concert.experiments.imaging.GratingInterferometryStepping.")
        self._output_directory = output_directory
        self._consumers = {}
        self._experiment = experiment
        self._dark_image = None
        self._reference_stepping = []
        self._object_stepping = []
        self.object_intensity = None
        self.object_phase = None
        self.object_visibility = None
        self.reference_intensity = None
        self.reference_phase = None
        self.reference_visibility = None
        self.intensity = None
        self.diff_phase = None
        self.visibility_contrast = None
        self.diff_phase_in_rad = None
        super().__init__(experiment.acquisitions)

    def _attach(self):
        self._consumers[self._experiment.get_acquisition("darks")] = self.process_darks
        self._consumers[self._experiment.get_acquisition(
            "reference_stepping")] = self.process_stepping
        self._consumers[self._experiment.get_acquisition(
            "object_stepping")] = self.process_stepping

        for acq, consumer in self._consumers.items():
            acq.consumers.append(consumer)

    def _detach(self):
        for acq, consumer in list(self._consumers.items()):
            acq.consumers.remove(consumer)

    async def process_darks(self, producer):
        """
        Processes dark images. All dark images are averaged.

        :param producer: Dark image producer
        :return:
        """
        self._dark_image = None
        async for item in producer:
            if self._dark_image is None:
                self._dark_image = item.astype(np.float32)
            else:
                self._dark_image += item
        self._dark_image /= await self._experiment.get_num_darks()

    async def process_stepping(self, producer):
        if await self._experiment.get_acquisition("reference_stepping").get_state() == "running":
            current_stepping = "reference"
            self._reference_stepping = []
        elif await self._experiment.get_acquisition("object_stepping").get_state() == "running":
            current_stepping = "object"
            self._object_stepping = []
        else:
            return

        async for item in producer:
            if current_stepping == "reference":
                self._reference_stepping.append(item)
            elif current_stepping == "object":
                self._object_stepping.append(item)
        if await self._experiment.acquisitions[-1].get_state() == "running":
            await self._compare_and_write()

    async def _process_data_and_write(self, stepping, dark_image, name):
        """
        :param stepping: Stepping data (list of images)
        :param dark_image: Averaged dark image
        :param name: Name of the stepping. If set to 'object' or 'reference', the resulting data is
            stored in the corresponding class variables.
        """
        intensity, phase, visibility = await self._process_data(stepping,
                                                                dark_image)
        if name == "object":
            self.object_intensity = intensity
            self.object_phase = phase
            self.object_visibility = visibility
        elif name == "reference":
            self.reference_intensity = intensity
            self.reference_phase = phase
            self.reference_visibility = visibility

        await self._write_single_image(f"{name}_intensity.tif", intensity)
        await self._write_single_image(f"{name}_phase.tif", phase)
        await self._write_single_image(f"{name}_visibility.tif", visibility)

    async def _compare_and_write(self):
        """
        Processes all acquired data and stores the resulting images.
        """
        if self._object_stepping:
            await self._process_data_and_write(self._object_stepping,
                                               self._dark_image,
                                               "object")

        if self._reference_stepping:
            await self._process_data_and_write(self._reference_stepping,
                                               self._dark_image,
                                               "reference")

        if self._reference_stepping and self._object_stepping:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.intensity = self.object_intensity / self.reference_intensity
                self.diff_phase = self.object_phase - self.reference_phase
                self.visibility_contrast = self.object_visibility / self.reference_visibility
            await self._write_single_image("intensity_contrast.tif", self.intensity)
            await self._write_single_image("visibility_contrast.tif", self.visibility_contrast)
            await self._write_single_image("differential_phase.tif", self.diff_phase)
            if self._experiment.get_propagation_distance() is not None:
                self.diff_phase_in_rad = (self.diff_phase
                                          * (await self._experiment.get_grating_period()
                                             / await self._experiment.get_propagation_distance()))
                await self._write_single_image("differential_phase_in_rad.tif",
                                               self.diff_phase_in_rad)

    async def _process_data(self, stepping, dark):
        stepping_curve = np.zeros((stepping[0].shape[0],
                                   stepping[0].shape[1],
                                   len(stepping)),
                                  dtype=np.float32)
        for i, step in enumerate(stepping):
            if dark is not None:
                stepping_curve[:, :, i] = step - dark
            else:
                stepping_curve[:, :, i] = step
        fft_object = np.fft.fft(stepping_curve)
        with np.errstate(divide='ignore', invalid='ignore'):
            fft_object = fft_object[:, :, (0, await self._experiment.get_num_periods())] / (
                await self._experiment.get_num_periods()
                * await self._experiment.get_num_steps_per_period())
            phase = np.angle(fft_object[:, :, 1])
            visibility = (2. * np.absolute(fft_object[:, :, 1])) / np.real(fft_object[:, :, 0])
            intensity = np.abs(np.real(fft_object[:, :, 0]))
        return intensity, phase, visibility

    async def _write_single_image(self, name, image):
        async with self._experiment.walker:
            file_name = os.path.join(self._experiment.walker.current, name)

        im_writer = self._experiment.walker.writer(file_name, bytes_per_file=0)
        im_writer.write(image)


class PCOTimestampCheck(Addon):
    def __init__(self, experiment):
        self._timestamp_checks = {}
        self._experiment = experiment
        self.timestamp_incorrect = False
        self.timestamp_missing = False
        super().__init__(experiment.acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        for acq in self.acquisitions:
            self._timestamp_checks[acq] = self._check_timestamp
            acq.consumers.append(self._timestamp_checks[acq])

    def _detach(self):
        """Detach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self._timestamp_checks)

    async def _check_timestamp(self, producer):
        self.timestamp_incorrect = False
        self.timestamp_missing = False
        i = 0
        last_acquisition = await self._experiment.acquisitions[-1].get_state() == "running"
        async for img in producer:
            if i == 0:
                if not isinstance(img, ImageWithMetadata) or (
                        isinstance(img, ImageWithMetadata) and 'frame_number' not in img.metadata):
                    self._experiment.log.error("No 'frame_number' present in image."
                                               "camera.timestamp needs to be set to 'both' or"
                                               "'binary' to use this addon."
                                               "Works only with pco cameras.")
                    self.timestamp_missing = True
                    return
            if img.metadata['frame_number'] != i + 1:
                self._experiment.log.error(
                    f"Frame {i + 1} had wrong frame number {img.metadata['frame_number']}.")
                self.timestamp_incorrect = True
            i += 1
        if last_acquisition and self.timestamp_incorrect:
            raise PCOTimestampCheckError("Not all 'frame_numbers' where correct.")
        if last_acquisition and self.timestamp_missing:
            raise PCOTimestampCheckError("Not all images contained timestamps.")


class AddonError(Exception):
    """Addon errors."""

    pass


class OnlineReconstructionError(Exception):
    pass


class PCOTimestampCheckError(Exception):
    pass
