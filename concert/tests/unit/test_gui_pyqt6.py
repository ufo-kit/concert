"""PyQt6 compatibility smoke tests, isolated from the core test process."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(module) is None for module in ('PyQt6', 'qasync')),
    reason='Install the gui extra to run PyQt6 compatibility tests',
)


def run_gui_script(script):
    # QApplication and Qt binding selection are process-global. Subprocesses
    # keep these tests independent of other viewers and asyncio test loops.
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM='offscreen', PYQTGRAPH_QT_LIB='PyQt6', QT_API='pyqt6')
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_widget_construction_and_async_slots():
    run_gui_script('''
import asyncio
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication
from concert.base import Parameterizable, Parameter, Quantity, Selection, State
from concert.quantities import q
from concert.gui import get_application
from concert.gui.parameterizable import ParameterizableWidget
from qasync import QEventLoop
import pyqtgraph

class Device(Parameterizable):
    number = Parameter()
    position = Quantity(q.mm)
    choice = Selection(['a', 'b'])
    state = State(default='standby')

    async def __ainit__(self):
        await super().__ainit__()
        self._number = 1
        self._position = 2 * q.mm
        self._choice = 'a'

    async def _get_number(self): return self._number
    async def _set_number(self, value): self._number = value
    async def _get_position(self): return self._position
    async def _set_position(self, value): self._position = value
    async def _get_choice(self): return self._choice
    async def _set_choice(self, value): self._choice = value

app = get_application()
assert isinstance(app, QApplication)
assert pyqtgraph.Qt.QT_LIB == 'PyQt6'
with QEventLoop(app) as loop:
    asyncio.set_event_loop(loop)
    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device)
    assert widget.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    assert widget.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert not widget.params['state'].write_button.isEnabled()

    async def exercise_slots():
        number = widget.params['number']
        await number.read()
        assert number.value.text() == '1'
        number.value.setText('7')
        await number.write()
        assert await device.get_number() == 7
        position = widget.params['position']
        position.value.setText('3 mm')
        await position.write()
        assert await device.get_position() == 3 * q.mm
        choice = widget.params['choice']
        choice.value.setCurrentIndex(1)
        await choice.write()
        assert await device.get_choice() == 'b'
        await choice.read()
        assert choice.value.currentText() == 'b'

    loop.run_until_complete(exercise_slots())
    number = widget.params['number']
    QTimer.singleShot(0, number.context_menu.close)
    number.context_menu.exec(number.mapToGlobal(number.rect().center()))
    for row in widget.params.values():
        row._timer.stop()
    widget.close()
''')


def test_gui_event_processing():
    run_gui_script('''
import asyncio
from concert.gui import get_application, gui_event_loop

app = get_application()

async def main():
    task = gui_event_loop()
    await asyncio.sleep(0.03)
    assert not task.done()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(main())
''')


def test_standalone_imports_use_pyqt6():
    run_gui_script('''
import runpy
from PyQt6.QtWidgets import QApplication
import qasync

namespace = runpy.run_path('bin/gui_standalone.py', run_name='gui_smoke_test')
assert namespace['QApplication'] is QApplication
assert qasync.QApplication is QApplication
''')


def test_async_operations_conversion_and_cleanup():
    run_gui_script('''
import asyncio
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QCheckBox
from concert.base import Parameterizable, Parameter, Selection
from concert.gui import get_application
from concert.gui.parameterizable import ParameterizableWidget
from qasync import QEventLoop

class Device(Parameterizable):
    value = Parameter()
    flag = Parameter()
    unsupported = Parameter()
    choice = Selection([None, 'a', ''])

    async def __ainit__(self):
        await super().__ainit__()
        self.calls = 0
        self.writes = []
        self.block = None
        self.fail = False
        self._value = 1
        self._flag = True
        self._choice = None

    async def _get_value(self):
        self.calls += 1
        if self.block is not None:
            await self.block.wait()
        if self.fail:
            raise ValueError('read failed')
        return self._value
    async def _set_value(self, value): self.writes.append(value)
    async def _get_flag(self): return self._flag
    async def _set_flag(self, value): self._flag = value
    async def _get_unsupported(self): return [1, 2]
    async def _set_unsupported(self, value): raise AssertionError('must be read-only')
    async def _get_choice(self): return self._choice
    async def _set_choice(self, value): self.writes.append(value); self._choice = value

async def main():
    device = await Device()
    widget = ParameterizableWidget(device)
    assert device.calls == 0
    assert widget.widget() is not None and widget.widgetResizable()
    row = widget.params['value']
    assert not row.write_button.isEnabled()
    events = []
    row.getter_started.connect(lambda: events.append('start'))
    row.getter_error.connect(lambda error: events.append('error'))
    row.getter_finished.connect(lambda: events.append('finish'))
    await row.read()
    assert device.calls == 1 and events == ['start', 'finish']
    device.fail = True
    events.clear()
    await row.read()
    assert events == ['start', 'error', 'finish']
    assert row.error_label.text() == 'read failed'
    device.fail = False
    device.block = asyncio.Event()
    before = device.calls
    task = asyncio.create_task(row.read())
    while device.calls == before:
        await asyncio.sleep(0)
    for _ in range(10):
        row.request_refresh()
        row.request_write()
        await row.read()
        await row.write()
    assert device.calls == before + 1 and device.writes == []
    assert not row.write_button.isEnabled()
    device.block.set()
    await task
    assert row.write_button.isEnabled()
    row.set_polling(True)
    assert row._timer.isActive() and row.poll_action.isChecked()
    row.show()
    row.hide()
    assert not row._timer.isActive() and not row.polling
    assert widget.params['value'] is row
    flag = widget.params['flag']
    await flag.read()
    assert isinstance(flag.value, QCheckBox)
    flag.value.setChecked(False)
    await flag.write()
    assert device._flag is False
    unsupported = widget.params['unsupported']
    await unsupported.read()
    assert not unsupported.write_button.isEnabled()
    selection = widget.params['choice']
    await selection.read()
    assert selection.value.currentIndex() == 0
    device.writes.clear()
    await selection.write()
    assert device.writes == [None]
    selection.value.setCurrentIndex(2)
    await selection.write()
    assert device.writes == [None, '']
    selection.value.setCurrentIndex(-1)
    await selection.write()
    assert device.writes == [None, '']
    device.block = asyncio.Event()
    row.request_refresh()
    while not row._busy:
        await asyncio.sleep(0)
    await widget.shutdown()
    assert not row._busy and row._task is None and not row._timer.isActive()
    widget.close()
    second = ParameterizableWidget(device, exclude_properties=['flag'])
    assert 'flag' not in second.params
    doomed = second.params['value']
    doomed.request_refresh()
    while not doomed._busy:
        await asyncio.sleep(0)
    doomed.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert sip.isdeleted(doomed) and 'value' not in second.params
    await second.shutdown()
    second.close()

app = get_application()
with QEventLoop(app) as loop:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
''')


def test_alignment_cancel_offsets_and_camera_restoration():
    run_gui_script('''
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import numpy as np
from PyQt6.QtCore import Qt, QPointF
from concert.quantities import q
from concert.gui import get_application
from concert.gui.tools import Alignment

class Camera:
    def __init__(self):
        self.source = 'EXTERNAL'
        self.state = 'recording'
        self.fail = False
    async def get_trigger_source(self): return self.source
    async def set_trigger_source(self, value): self.source = value
    async def get_state(self): return self.state
    async def stop_recording(self): self.state = 'standby'
    async def start_recording(self): self.state = 'recording'
    async def trigger(self): pass
    async def grab(self):
        if self.fail: raise ValueError('grab failed')
        return np.zeros((10, 10))
    @asynccontextmanager
    async def recording(self):
        await self.start_recording()
        try: yield
        finally: await self.stop_recording()

async def main():
    camera = Camera()
    motors = [SimpleNamespace(move=AsyncMock(), set_position=AsyncMock()) for _ in range(4)]
    viewer = Mock()
    viewer.getImageItem.return_value.mapFromScene.side_effect = lambda pos: pos
    with patch('concert.gui.tools.ImageView', return_value=viewer):
        align = await Alignment(*motors, camera, 2 * q.mm,
                                target_horizontal=8, target_vertical=6)
    assert align._target_horizontal == 8 and align._target_vertical == 6
    def click(button):
        align.mouse_clicked(SimpleNamespace(button=lambda: button, scenePos=lambda: QPointF(3, 4)))
    viewer.show.side_effect = lambda: click(Qt.MouseButton.RightButton)
    assert await align() is False
    for motor in motors[:3]: motor.move.assert_not_awaited()
    assert motors[3].set_position.await_count == 1  # no second view after cancellation
    assert camera.source == 'EXTERNAL' and camera.state == 'recording'
    viewer.hide.assert_called()
    viewer.show.side_effect = lambda: click(Qt.MouseButton.LeftButton)
    assert await align.align_x(z=True)
    motors[0].move.assert_awaited_once_with(10 * q.mm)
    motors[2].move.assert_awaited_once_with(-4 * q.mm)
    camera.fail = True
    try: await align.align_y()
    except ValueError: pass
    else: raise AssertionError('grab error must propagate')
    assert camera.source == 'EXTERNAL' and camera.state == 'recording'
    camera.fail = False
    ready = asyncio.Event()
    viewer.show.side_effect = ready.set
    task = asyncio.create_task(align.align_y())
    await ready.wait()
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    else: raise AssertionError('cancellation must propagate')
    assert camera.source == 'EXTERNAL' and camera.state == 'recording'
    motors[1].move.assert_not_awaited()
    align._input_event.clear()
    click(Qt.MouseButton.MiddleButton)
    assert not align._input_event.is_set()

get_application()
asyncio.run(main())
''')
