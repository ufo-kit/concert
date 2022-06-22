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
from PyQt5.QtWidgets import QPushButton, QWidget, QHBoxLayout, QGroupBox
import asyncio
import sys


async def main():
    def close_future(future, loop):
        loop.call_later(10, future.cancel)
        future.cancel()

    loop = asyncio.get_event_loop()
    future = asyncio.Future()

    app = QApplication.instance()

    lin = await LinearMotor()
    lin_widget = ParameterizableWidget(lin)
    lin_widget.setWindowTitle("LinMotor")
    lin_widget.update()

    sample_changer = await SampleChanger()
    sample_changer_widget = ParameterizableWidget(sample_changer)
    sample_changer_widget.setWindowTitle("Sample Changer")
    sample_changer_widget.update()

    # Experiment
    walker = DirectoryWalker()
    camera = await Camera()
    shutter = await Shutter()
    tomo_motor = await RotationMotor()
    tomo_motor_widget = ParameterizableWidget(tomo_motor)
    tomo_motor_widget.setWindowTitle("Tomo motor")
    exp = await SteppedTomography(walker=walker, flat_motor=lin, tomography_motor=tomo_motor, radio_position=0 * q.mm,
                                  flat_position=10 * q.mm, camera=camera, shutter=shutter)
    viewer = await PyQtGraphViewer()

    live_view = Consumer(exp.acquisitions, viewer)
    exp_widget = ParameterizableWidget(exp)
    exp_widget.setWindowTitle("Experiment")
    run_button = QPushButton("run experiment")

    @asyncSlot()
    async def run_exp(a):
        await exp.run()

    run_button.clicked.connect(run_exp)
    exp_widget._layout.addWidget(run_button)
    exp_widget.update()

    def make_frame(name, widget):
        group_box = QGroupBox(name)
        _layout = QHBoxLayout()
        _layout.addWidget(widget)
        group_box.setLayout(_layout)
        return group_box

    main_window = QWidget()
    layout = QHBoxLayout()
    layout.addWidget(make_frame("Tomo motor", tomo_motor_widget))
    layout.addWidget(make_frame("Sample changer", sample_changer_widget))
    layout.addWidget(make_frame("Lin motor", lin_widget))
    layout.addWidget(make_frame("CT Experiment", exp_widget))

    main_window.setLayout(layout)
    main_window.show()

    await future
    return True


if __name__ == "__main__":
    try:
        qasync.run(main())
    except asyncio.exceptions.CancelledError:
        sys.exit(0)
