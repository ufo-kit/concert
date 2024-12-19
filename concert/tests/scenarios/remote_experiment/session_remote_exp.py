import asyncio
import os
import zmq
import concert
from concert.experiments.addons import tango as tango_addons
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.shutters.dummy import Shutter
from concert.storage import RemoteDirectoryWalker
from concert.networking.base import get_tango_device
from concert.experiments.synchrotron import RemoteContinuousTomography
from concert.devices.cameras.uca import RemoteNetCamera
from concert.helpers import CommData

####################################################################################################
# Docker daemon creates a DNS entry inside the specified network with the service name. We need to 
# specify this domain name to communicate with a service on a given exposed port. In the compose.yml
# we have specified `uca_camera` and `remote_walker` as service names for the mock camera and walker
# tango servicer processes running inside their respective containers. Hence, in the session we'd
# have to use these domain names.
####################################################################################################

walker_dev_uri = f"remote_walker:7001/concert/tango/walker#dbase=no"

# Experimental configurations
num_darks = 10
num_flats = 10
num_radios = 100

# Communications Metadata
SERVERS = {
        "walker": CommData("uca_camera", port=8993, socket_type=zmq.PUSH),
}

# Camera
camera = await RemoteNetCamera()
if await camera.get_state() == 'recording':
    await camera.stop_recording()

# Walker | Writer Configuration
root = "/mnt/ips_image_mnt"
walker_device = get_tango_device(walker_dev_uri, timeout=30 * 60 * q.s)
walker = await RemoteDirectoryWalker(device=walker_device, root=root, bytes_per_file=2**40)

# Experiment Configuration
shutter = await Shutter()
flat_motor = await LinearMotor()
tomo = await ContinuousRotationMotor()
exp = await RemoteContinuousTomography(walker=walker,
                                       flat_motor=flat_motor, 
                                       tomography_motor=tomo,
                                       radio_position=0*q.mm,
                                       flat_position=10*q.mm,
                                       camera=camera,
                                       shutter=shutter,
                                       num_flats=num_flats,
                                       num_darks=num_darks,
                                       num_projections=num_radios)
_ = await tango_addons.ImageWriter(exp, SERVERS["walker"], exp.acquisitions)

# Run Experiment
_ = await exp.run()
