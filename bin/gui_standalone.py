from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.cameras.dummy import Camera
from concert.devices.samplechangers.dummy import SampleChanger
from concert.experiments.synchrotron import LocalSteppedTomography
from concert.devices.shutters.dummy import Shutter
from concert.experiments.addons.local import Consumer
from concert.ext.viewers import PyQtGraphViewer
from concert.storage import DirectoryWalker
from concert.gui.parameterizable import ParameterizableWidget
from concert.gui import get_application, GuiSession
from concert.quantities import q
from PyQt6.QtWidgets import (QApplication, QPushButton, QWidget, QHBoxLayout, QGroupBox,
                             QVBoxLayout, QTabWidget, QLabel)
from concert.gui.style import apply_style
from qasync import asyncSlot

import qasync
import asyncio
import sys


async def main(auto_close=False):
    app = get_application()
    # Keep Qt running until async cleanup completes, rather than awaiting it
    # after aboutToQuit has already stopped the qasync event loop.
    app.setQuitOnLastWindowClosed(False)
    session = GuiSession().start()

    lin = await LinearMotor()
    lin_widget = ParameterizableWidget(lin, session=session, order=['position', 'state'])
    lin_widget.setWindowTitle("LinMotor")

    sample_changer = await SampleChanger()
    sample_changer_widget = ParameterizableWidget(sample_changer, session=session)
    sample_changer_widget.setWindowTitle("Sample Changer")

    # Experiment
    walker = await DirectoryWalker()
    camera = await Camera()
    shutter = await Shutter()
    tomo_motor = await RotationMotor()
    tomo_motor_widget = ParameterizableWidget(tomo_motor, session=session, order=['position', 'state'])
    tomo_motor_widget.setWindowTitle("Tomo motor")
    camera_widget = ParameterizableWidget(camera, session=session, groups={
        'Acquisition': ['exposure_time', 'frame_rate'],
        'Image size': ['roi_width', 'roi_height'],
    })

    exp = await LocalSteppedTomography(walker=walker, flat_motor=lin, tomography_motor=tomo_motor, radio_position=0 * q.mm,
                                  flat_position=10 * q.mm, camera=camera, shutter=shutter)


    viewer = await PyQtGraphViewer()
    live_view = await Consumer(experiment=exp, consumer=viewer)
    exp_widget = ParameterizableWidget(exp, session=session)
    exp_widget.setWindowTitle("Experiment")
    run_button = QPushButton('Run experiment')
    run_button.setObjectName('concertPrimary')
    run_button.setMinimumHeight(40)



    experiment_task = None

    @asyncSlot(bool)
    async def run_exp(_checked=False):
        nonlocal experiment_task
        experiment_task = asyncio.current_task()
        run_button.setEnabled(False)
        try:
            await exp.run()
        finally:
            run_button.setEnabled(True)
            experiment_task = None

    run_button.clicked.connect(run_exp)
    exp_widget.add_widget(run_button)

    def make_frame(name, widget):
        group_box = QGroupBox(name)
        group_box.setObjectName('concertGroup')
        _layout = QHBoxLayout()
        _layout.setContentsMargins(0, 12, 0, 0)
        _layout.addWidget(widget)
        group_box.setLayout(_layout)
        return group_box

    closed = asyncio.Event()

    class MainWindow(QWidget):
        def closeEvent(self, event):
            # Closing the UI is not an experiment abort.
            run_button.setEnabled(False)
            closed.set()
            super().closeEvent(event)

    main_window = MainWindow()
    main_window.setObjectName('concertWindow')
    main_window.setWindowTitle('Concert · Device control')
    main_window.resize(960, 720)
    apply_style(main_window)
    layout = QVBoxLayout()
    layout.setContentsMargins(24, 20, 24, 20)
    layout.setSpacing(12)
    title = QLabel('Device control')
    title.setObjectName('concertTitle')
    subtitle = QLabel('Concert · Dummy devices and tomography experiment')
    subtitle.setObjectName('concertSubtitle')
    layout.addWidget(title)
    layout.addWidget(subtitle)
    tabs = QTabWidget()
    motors = QWidget()
    motor_layout = QHBoxLayout(motors)
    motor_layout.addWidget(make_frame('Rotation motor', tomo_motor_widget))
    motor_layout.addWidget(make_frame('Linear motor', lin_widget))
    tabs.addTab(motors, 'Motors')
    tabs.addTab(camera_widget, 'Camera')
    tabs.addTab(sample_changer_widget, 'Sample changer')
    tabs.addTab(exp_widget, 'Experiment')
    layout.addWidget(tabs, 1)

    main_window.setLayout(layout)
    main_window.show()

    if auto_close:
        asyncio.get_running_loop().call_soon(main_window.close)
    try:
        await closed.wait()
    finally:
        if experiment_task is not None:
            await asyncio.shield(experiment_task)
        await asyncio.gather(*(widget.shutdown() for widget in (
            lin_widget, sample_changer_widget, tomo_motor_widget, camera_widget, exp_widget)))
        await session.shutdown()


if __name__ == "__main__":
    get_application()
    try:
        qasync.run(main())
    except asyncio.exceptions.CancelledError:
        sys.exit(0)
