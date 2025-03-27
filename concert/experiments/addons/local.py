import os
import numpy as np
from concert.coroutines.base import async_generate
from concert.coroutines.sinks import Accumulate
from concert.experiments.addons import base
from concert.experiments.base import Consumer as AcquisitionConsumer, local
from concert.experiments.imaging import LocalGratingInterferometryStepping
from concert.helpers import PerformanceTracker, ImageWithMetadata
from concert.quantities import q


class Benchmarker(base.Benchmarker):
    async def __ainit__(self, experiment, acquisitions=None):
        await base.Benchmarker.__ainit__(self, experiment=experiment, acquisitions=acquisitions)
        self._durations = {}

    @local
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


class ImageWriter(base.ImageWriter):
    async def __ainit__(self, experiment, acquisitions=None):
        await base.ImageWriter.__ainit__(self, experiment=experiment, acquisitions=acquisitions)

    @local
    async def write_sequence(self, name, producer=None):
        """Wrap the walker and write data to subdirectory *name*."""
        return await self.walker.create_writer(producer, name=name)


class Consumer(base.Consumer):
    async def __ainit__(self, consumer, experiment, acquisitions=None):
        await base.Consumer.__ainit__(self,
                                      consumer=consumer,
                                      experiment=experiment,
                                      acquisitions=acquisitions)

    @local
    async def consume(self, producer):
        await self._consumer(producer)


class LiveView(base.LiveView):
    async def __ainit__(self, viewer, experiment, acquisitions=None):
        await base.LiveView.__ainit__(self,
                                      viewer,
                                      experiment=experiment,
                                      acquisitions=acquisitions)

    @local
    async def consume(self, producer):
        await self._viewer(producer)


class Accumulator(base.Accumulator):
    async def __ainit__(self, experiment, acquisitions=None, shapes=None, dtype=None):
        await base.Accumulator.__ainit__(
            self,
            experiment=experiment,
            acquisitions=acquisitions,
            shapes=shapes,
            dtype=dtype
        )
        self._accumulators = {}

    @local
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


class OnlineReconstruction(base.OnlineReconstruction):
    async def __ainit__(self, experiment, acquisitions=None, do_normalization=True,
                        average_normalization=True, slice_directory='online-slices',
                        viewer=None):
        from concert.ext.ufo import LocalGeneralBackprojectArgs

        await base.OnlineReconstruction.__ainit__(
            self,
            LocalGeneralBackprojectArgs(),
            experiment=experiment,
            acquisitions=acquisitions,
            do_normalization=do_normalization,
            average_normalization=average_normalization,
            slice_directory=slice_directory,
            viewer=viewer
        )
        from concert.ext.ufo import GeneralBackprojectManager

        self._manager = await GeneralBackprojectManager(
            self.args,
            average_normalization=average_normalization
        )

    @local
    async def update_darks(self, producer):
        return await self._manager.update_darks(producer)

    @local
    async def update_flats(self, producer):
        return await self._manager.update_flats(producer)

    async def _reconstruct(self, producer=None, slice_directory=None):
        if producer is None:
            await self._manager.backproject(async_generate(self._manager.projections))
        else:
            await self._manager.backproject(producer)

        if self.walker:
            if (
                producer is not None and await self.get_slice_directory()
                or producer is None and slice_directory
            ):
                async with self.walker:
                    producer = async_generate(self._manager.volume)
                    writer = self.walker.create_writer(
                        producer,
                        name=await self.get_slice_directory() if slice_directory is None else slice_directory,
                        dsetname='slice_{:>04}.tif'
                    )
                await writer

    @local
    async def reconstruct(self, producer):
        await base.OnlineReconstruction.reconstruct(self, producer=producer)

    async def _rereconstruct(self, slice_directory=None):
        await self._reconstruct(producer=None, slice_directory=slice_directory)

    async def find_parameter(self, parameter, region, metric='sag', z=None, store=False):
        # Unit conversion and simple list creation
        region = region.to(self.UNITS[parameter.replace('-', '_')]).magnitude.tolist()

        return (
            await self._manager.find_parameters(
                [parameter],
                regions=[region],
                metrics=[metric],
                store=store,
                z=0 if z is None else z.magnitude,
            )
        )[0]

    async def get_volume(self):
        return self._manager.volume

    async def _get_slice_x(self, index):
        return self._manager.volume[:, :, index]

    async def _get_slice_y(self, index):
        return self._manager.volume[:, index, :]

    async def _get_slice_z(self, index):
        return self._manager.volume[index]


class PhaseGratingSteppingFourierProcessing(base.PhaseGratingSteppingFourierProcessing):
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

    @local
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

    @local
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
            file_name = os.path.join(await self._experiment.walker.get_current(), name)

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


class PCOTimestampCheck(base.Addon):

    async def __ainit__(self, experiment, acquisitions=None):
        self._timestamp_checks = {}
        self._experiment = experiment
        self.timestamp_incorrect = False
        self.timestamp_missing = False
        await super().__ainit__(experiment=experiment, acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        """Attach all acquisitions."""
        consumers = {}

        for acq in acquisitions:
            consumers[acq] = AcquisitionConsumer(self._check_timestamp)

        return consumers

    @local
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


class PCOTimestampCheckError(Exception):
    pass
