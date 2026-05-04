.. _setup:

===================================
Setting up a beamline or laboratory
===================================

This will give some guide and good practices for setting up a beamline or laboratory with Concert.
This will cover how to implement devices, organize them in a session and to create custom experiments.
We assume that you already have installed concert.

Implementing devices
====================

Concert features abstract base classes for various types of devices, such as motors, cameras, detectors, etc to define the default interface to those.
To implement a device, you have to subclass the appropriate base class and implement the required methods.
For reference and testing there are also dummy devices available, which can be used as a template for your own implementation.

The most clean way to organize you implementations is to put them in a separate Python module, which can be imported from your session.
Or you can put them at a location where the session can find them, e.g. in the same directory as the session file.

Assuming you want to do stepped tomography, you need to implement a rotation stage, a linear stage (for flat field images), a shutter (or X-ray tube for a laboratory based setup) and a camera.
You have to implement all abstract methods of the base classes.

Shutter
_______
::

    from concert.devices.shutters.base import Shutter as BaseShutter

    class Shutter(BaseShutter):
        async def _open(self):
            # code to open the shutter
            pass

        async def _close(self):
            # code to close the shutter
            pass

        async def _get_state(self) -> str:
            # code to get the state of the shutter
            # Should return 'open' or 'closed'
            pass

Rotation Motor
______________
::

    from concert.devices.motors.base import RotationMotor as BaseRotationMotor
    from concert.quantities import q

    class RotationMotor(BaseRotationMotor):

        async def _get_position(self):
            # code to get the current position of the motor (with units)
            return 0.0 * q.deg

        async def _set_position(self, position):
            # code to set the position of the motor (with units)
            # It must block until the motor has reached the target position
            pass

        async def _get_state(self) -> str:
            # code to get the state of the motor
            # Should return 'moving' or 'standby'
            pass


Linear Motor
____________

This is the same as the rotation motor, but with different units and a different base class::

    from concert.devices.motors.base import LinearMotor as BaseLinearMotor
    from concert.quantities import q

    class LinearMotor(BaseLinearMotor):

        async def _get_position(self):
            # code to get the current position of the motor (with units)
            return 0.0 * q.mm

        async def _set_position(self, position):
            # code to set the position of the motor (with units)
            # It must block until the motor has reached the target position
            pass

        async def _get_state(self) -> str:
            # code to get the state of the motor
            # Should return 'moving' or 'standby'
            pass

Camera
_______________

The camera will create the images::

    from concert.devices.cameras.base import Camera as BaseCamera
    from concert.quantities import q

    class Camera(BaseCamera):

        async def _get_trigger_source(self) -> str:
            # Returns the current trigger source of the camera, e.g. 'EXTERNAL' or, 'AUTO' or 'SOFTWARE'
            # 'AUTO' means that the camera triggers itself, e.g. with an internal timer. A 'grab()' call will always return the next finished frame.
            # 'SOFTWARE' means that the camera will only trigger when a 'trigger()' call is made. A 'grab()' call will return the current frame. Grab will block until a frame is available, but it will not trigger the camera.
            # 'EXTERNAL' the same as 'SOFTWARE', but the camera will be triggered by an external signal, e.g. from a function generator or a shutter.
            pass

        async def _set_trigger_source(self, source):
            # Sets the trigger source of the camera, e.g. 'EXTERNAL' or, 'AUTO' or 'SOFTWARE'
            pass

        async def _record_real(self):
            # Code to start the recording of the camera. This should block until the recording has started.
            pass

        async def _stop_real(self):
            # Code to stop the recording of the camera. This should block until the recording has stopped.
            pass

        async def _trigger_real(self):
            # Code to trigger the camera. This should block until the camera has been triggered.
            pass

        async def _grab_real(self) -> ImageWithMetadata:
            # Code to grab a frame from the camera. This should block until a frame is available and return an ImageWithMetadata object.
            pass

        async def _get_state(self) -> str:
            # code to get the state of the camera
            # Should return 'recording' or 'standby'
            pass

        async def _get_frame_rate(self):
            # code to get the current frame rate of the camera with units
            return 0.0 * q.Hz

        async def _set_frame_rate(self, frame_rate):
            # code to set the frame rate of the camera with units
            pass

After implementing these devices you can import them in your session and create instances of them, which will be available as devices in your session. You can also create custom experiments, which will be available as functions in your session.

Organizing devices in sessions
================================

An easy way to organize your devices is to create instance of them in a sub-module and import them in cour sessions.

E.g. create a subfolder 'devices' in you session folder and create files like 'optics.py' and 'tomography_station.py' there.::

    # devices/optics.py
    from myconcertdevices.shutters import Shutter

    shutter = await Shutter()

    # devices/tomography_station.py
    from myconcertdevices.motors import RotationMotor, LinearMotor
    from myconcertdevices.cameras import Camera
    tomo_motor = await RotationMotor()
    flat_motor = await LinearMotor()
    camera = await Camera()


Now you can import these devices in your session and they will be available as devices in your session.::

    from devices.optics import shutter
    from devices.tomography_station import tomo_motor, flat_motor


Building a session for tomography
=================================

Now we create a session for doing tomography. We will create a session called 'tomography' and add the devices we just created to it. We will also create an experiment for doing a scan.

The session would look like this::

    from devices.optics import shutter
    from devices.tomography_station import tomo_motor, flat_motor, camera

    from concert.experiment.sychrotron import LocalSteppedTomography
    from concert.storage import DirectoryWalker
    from concert.ext.viewers import PyqtgraphViewer

    # The walker handles the storage of the data. It will create a new directory for each experiment and store the data there.
    walker = await DirectoryWalker(root='/path/to/data')

    # The viewer is used for viewing the camera images
    viewer = await PyqtgraphViewer()
    # with the camera implemented you can already do viewer(camera.stream()) to see live images from the camera.


    # The experimment organizes the data acquisitions.

    experiment = await LocalSteppedTomography(walker=walker, flat_motor=flat_motor, tomography_motor=tomo_motor,
                        radio_position=0*q.mm, flat_position=20*q.mm, camera=camera, shutter=shutter, num_flats=200,
                        num_darks=200, num_projections=3000, angular_range=180 * q.deg, start_angle=0 * q.deg)

    # Addons are used to handle the created images. Here we use live view and image writer:
    from concert.experiments.addons.local import LiveView, ImageWriter
    live_view = await LiveView(viewer=viewer, experiment=experiment)
    image_writer = await ImageWriter(experiment=experiment)

With this session you only have to type 'await experiment.run()' to start the tomography scan. The data will be stored in the specified directory and you can view the camera images live in the viewer.

Implementing custom experiments
=================================

You can either create a subclass of en existing experiment (see API docs) or create a new experiment from scratch.
All experiments are composed of a sequence of Acquisitions. (They also can be added and removed during runtime, which allows for dynamic experiments, e.g. for adaptive scanning).
Each acquisition needs a 'generator' function, which is a coroutine that yields the acquired data. The experiment will run the generator and handle the data according to the addons that are added to the experiment.

The most simple experiment would look like this::

    from concert.experiments.base import Experiment, local, Acquisition

    class MyExperiment(Experiment):
        async def __ainit__(self, camera, walker):
            self.camera = camera
            frame_acquisition = await Acquisition(generator=self.generator, name='frame')
            await Experiment.__ainit__(self, acquisitions=[frame_acquisition] walker)

        @local
        async def generator(self):
            num_images = 100
            for i in range(num_images):
                # code to acquire an image, e.g. from a camera
                image = await camera.grab()
                yield image

This would create 100 images from the camera and store them in the directory specified by the walker.