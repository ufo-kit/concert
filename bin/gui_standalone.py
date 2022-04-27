from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.cameras.dummy import Camera
from concert.devices.samplechangers.dummy import SampleChanger
from concert.experiments.synchrotron import SteppedTomography
from concert.devices.shutters.dummy import Shutter
from concert.experiments.addons import Consumer
from concert.ext.viewers import PyQtGraphViewer
from concert.storage import DirectoryWalker
from concert.gui.parameterizable import ParameterizableWidget
from concert.quantities import q
from qasync import QApplication, asyncSlot

import qasync
from PyQt5.QtWidgets import QPushButton
import asyncio
import sys


async def main():
    def close_future(future, loop):
        loop.call_later(10, future.cancel)
        future.cancel()

    loop = asyncio.get_event_loop()
    future = asyncio.Future()

    app = QApplication.instance()

    lin = LinearMotor()
    lin_widget = ParameterizableWidget(lin)
    lin_widget.show()
    lin_widget.setWindowTitle("LinMotor")
    lin_widget.update()

    sample_changer = SampleChanger()
    sample_changer_widget = ParameterizableWidget(sample_changer)
    sample_changer_widget.setWindowTitle("Sample Changer")
    sample_changer_widget.show()
    sample_changer_widget.update()

    # Experiment
    walker = DirectoryWalker()
    camera = Camera()
    shutter = Shutter()
    tomo_motor = RotationMotor()
    tomo_motor_widget = ParameterizableWidget(tomo_motor)
    tomo_motor_widget.setWindowTitle("Tomo motor")
    tomo_motor_widget.show()
    exp = SteppedTomography(walker=walker, flat_motor=lin, tomography_motor=tomo_motor, radio_position=0 * q.mm,
                            flat_position=10*q.mm, camera=camera, shutter=shutter)
    viewer = PyQtGraphViewer()
    live_view = Consumer(exp.acquisitions, viewer)
    exp_widget = ParameterizableWidget(exp)
    exp_widget.setWindowTitle("Experiment")
    run_button = QPushButton("run experiment")

    @asyncSlot()
    async def run_exp(a):
        await exp.run()

    run_button.clicked.connect(run_exp)
    exp_widget._layout.addWidget(run_button)
    exp_widget.show()
    exp_widget.update()

    await future
    return True


if __name__ == "__main__":
    try:
        qasync.run(main())
    except asyncio.exceptions.CancelledError:
        sys.exit(0)
