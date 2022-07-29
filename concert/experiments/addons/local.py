import os
import numpy as np
from concert.coroutines.base import async_generate
from concert.coroutines.sinks import Accumulate
from concert.experiments.addons import base
from concert.experiments.imaging import LocalGratingInterferometryStepping
from concert.helpers import PerformanceTracker
from concert.quantities import q


class LocalMixin:
    """LocalMixin needs a producer becuase the backend processes image streams from concer."""

    remote = False


class Benchmarker(LocalMixin, base.Benchmarker):

    async def __ainit__(self, acquisitions=None):
        await base.Benchmarker.__ainit__(self, acquisitions=acquisitions)
        self._durations = {}

    async def start_timer(self, producer, acquisition_name):
        total_bytes = 0
        with PerformanceTracker() as pt:
            async for image in producer:
                total_bytes += image.nbytes
            pt.size = total_bytes * q.B
        self._durations[acquisition_name] = pt.duration

    async def _get_duration(self, acquisition_name):
        return self._durations[acquisition_name]

    async def _teardown(self):
        self._durations = {}


class ImageWriter(LocalMixin, base.ImageWriter):
    async def __ainit__(self, walker, acquisitions=None):
        await base.ImageWriter.__ainit__(self, walker, acquisitions=acquisitions)

    def write_sequence(self, name, producer=None):
        """Wrap the walker and write data to subdirectory *name*."""
        return self.walker.create_writer(producer, name=name)


class Consumer(LocalMixin, base.Consumer):
    async def __ainit__(self, consumer, acquisitions=None):
        await base.Consumer.__ainit__(self, consumer, acquisitions=acquisitions)

    async def consume(self, producer):
        await self._consumer(producer)


class LiveView(LocalMixin, base.LiveView):
    async def __ainit__(self, viewer, acquisitions=None):
        await base.LiveView.__ainit__(self, viewer, acquisitions=acquisitions)

    async def consume(self, producer):
        await self._viewer(producer)


class Accumulator(LocalMixin, base.Accumulator):
    async def __ainit__(self, acquisitions=None, shapes=None, dtype=None):
        await base.Accumulator.__ainit__(
            self,
            acquisitions=acquisitions,
            shapes=shapes,
            dtype=dtype
        )
        self._accumulators = {}

    def accumulate(self, producer, acquisition_name, shape=None, dtype=None):
        if acquisition_name not in self._accumulators:
            self._accumulators[acquisition_name] = Accumulate(shape=shape, dtype=dtype)

        return self._accumulators[acquisition_name](producer)

    async def _get_items(self, acquisition_name):
        if acquisition_name not in self._accumulators:
            return []
        return self._accumulators[acquisition_name].items

    async def teardown(self):
        self._accumulators = {}


class OnlineReconstruction(LocalMixin, base.OnlineReconstruction):
    async def __ainit__(self, acquisitions=None, do_normalization=True,
                        average_normalization=True, walker=None, slice_directory='online-slices'):
        await base.OnlineReconstruction.__ainit__(
            self,
            acquisitions=acquisitions,
            do_normalization=do_normalization,
            average_normalization=average_normalization,
            walker=walker,
            slice_directory=slice_directory
        )
        from concert.ext.ufo import GeneralBackprojectManager, QuantifiedArgs

        self._args = await QuantifiedArgs()
        self._manager = await GeneralBackprojectManager(
            self._args.tofu_args,
            average_normalization=average_normalization
        )

    async def update_darks(self, producer):
        return await self._manager.update_darks(producer)

    async def update_flats(self, producer):
        return await self._manager.update_flats(producer)

    async def reconstruct(self, producer):
        await self._manager.backproject(producer)
        if self.walker:
            async with self.walker:
                producer = async_generate(self._manager.volume)
                writer = self.walker.create_writer(
                    producer,
                    name=self.slice_directory,
                    dsetname='slice_{:>04}.tif'
                )
            await writer

    async def get_volume(self):
        return self._manager.volume

    async def _get_slice_x(self, index):
        return self._manager.volume[:, :, index]

    async def _get_slice_y(self, index):
        return self._manager.volume[:, index, :]

    async def _get_slice_z(self, index):
        return self._manager.volume[index]


class PhaseGratingSteppingFourierProcessing(LocalMixin, base.PhaseGratingSteppingFourierProcessing):
    async def __ainit__(self, experiment, output_directory="contrasts"):
        if not isinstance(experiment, LocalGratingInterferometryStepping):
            raise Exception("This addon can only be used with "
                            "concert.experiments.imaging.GratingInterferometryStepping.")
        self._output_directory = output_directory
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
        await base.PhaseGratingSteppingFourierProcessing.__ainit__(
            self,
            experiment,
            output_directory=output_directory
        )

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

    async def get_object_intensity(self):
        return self.object_intensity

    async def get_object_phase(self):
        return self.object_phase

    async def get_object_visibility(self):
        return self.object_visibility

    async def get_reference_intensity(self):
        return self.reference_intensity

    async def get_reference_phase(self):
        return self.reference_phase

    async def get_reference_visibility(self):
        return self.reference_visibility

    async def get_intensity(self):
        return self.intensity

    async def get_diff_phase(self):
        return self.diff_phase

    async def get_visibility_contrast(self):
        return self.visibility_contrast

    async def get_diff_phase_in_rad(self):
        return self.diff_phase_in_rad
