from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.cameras.dummy import Camera
from concert.devices.samplechangers.dummy import SampleChanger
from concert.experiments.synchrotron import LocalSteppedTomography
from concert.devices.shutters.dummy import Shutter
from concert.experiments.addons.local import Consumer
from concert.ext.viewers import PyQtGraphViewer
from concert.storage import DirectoryWalker
from concert.gui.parameterizable import ParameterizableWidget
from concert.gui import get_application
from concert.quantities import q
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QGroupBox
from qasync import asyncSlot

import qasync
import asyncio
import sys


async def main():
    app = get_application()

    lin = await LinearMotor()
    lin_widget = ParameterizableWidget(lin)
    lin_widget.setWindowTitle("LinMotor")
    await lin_widget.refresh()

    sample_changer = await SampleChanger()
    sample_changer_widget = ParameterizableWidget(sample_changer)
    sample_changer_widget.setWindowTitle("Sample Changer")
    await sample_changer_widget.refresh()

    # Experiment
    walker = DirectoryWalker()
    camera = await Camera()
    shutter = await Shutter()
    tomo_motor = await RotationMotor()
    tomo_motor_widget = ParameterizableWidget(tomo_motor)
    tomo_motor_widget.setWindowTitle("Tomo motor")
    camera_widget = ParameterizableWidget(camera)

    exp = await LocalSteppedTomography(walker=walker, flat_motor=lin, tomography_motor=tomo_motor, radio_position=0 * q.mm,
                                  flat_position=10 * q.mm, camera=camera, shutter=shutter)


    viewer = await PyQtGraphViewer()
    live_view = await Consumer(experiment=exp, consumer=viewer)
    exp_widget = ParameterizableWidget(exp)
    exp_widget.setWindowTitle("Experiment")
    run_button = QPushButton("run experiment")



    @asyncSlot()
    async def run_exp():
        run_button.setEnabled(False)
        try:
            await exp.run()
        finally:
            run_button.setEnabled(True)

    run_button.clicked.connect(run_exp)
    exp_widget.add_widget(run_button)
    await exp_widget.refresh()
    await tomo_motor_widget.refresh()
    await camera_widget.refresh()

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
    layout.addWidget(make_frame("Camera", camera_widget))
    layout.addWidget(make_frame("CT Experiment", exp_widget))

    main_window.setLayout(layout)
    main_window.show()

    closed = asyncio.Event()
    app.aboutToQuit.connect(closed.set)
    try:
        await closed.wait()
    finally:
        await asyncio.gather(*(widget.shutdown() for widget in (
            lin_widget, sample_changer_widget, tomo_motor_widget, camera_widget, exp_widget)))


if __name__ == "__main__":
    get_application()
    try:
        qasync.run(main())
    except asyncio.exceptions.CancelledError:
        sys.exit(0)
