"""Normal pytest-qt tests for presentation; device logic is tested separately."""
import asyncio

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('qasync')
pytest.importorskip('pytestqt')

from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QApplication, QTabWidget, QWidget
from qasync import QEventLoop

from concert.base import Parameterizable, Parameter, Quantity, Selection
from concert.gui import GuiSession
from concert.gui.editors import ScalarEditor
from concert.gui.parameterizable import ParameterizableWidget
from concert.gui.tools import Alignment
from concert.quantities import q
from unittest.mock import patch


class Device(Parameterizable):
    value = Parameter(help='An editable scalar')
    position = Quantity(q.mm, lower=0 * q.mm, upper=10 * q.mm)

    async def __ainit__(self):
        await super().__ainit__()
        self.current = True

    async def _get_value(self):
        return self.current

    async def _set_value(self, value):
        self.current = value

    async def _get_position(self):
        return 2 * q.mm

    async def _get_target_position(self):
        return 4 * q.mm

    async def _set_position(self, value):
        pass


@pytest.fixture
def loop(qapp):
    with QEventLoop(qapp) as event_loop:
        asyncio.set_event_loop(event_loop)
        yield event_loop
    asyncio.set_event_loop(None)


def test_editor_changes_type_and_recovers(qtbot, loop):
    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device)
    qtbot.addWidget(widget)
    row = widget.params['value']
    assert row.name_label.toolTip() == 'An editable scalar'
    loop.run_until_complete(row.read())
    assert isinstance(row.value, QCheckBox)
    widget.show()
    qtbot.waitExposed(widget)
    loop.run_until_complete(asyncio.sleep(0.1))
    qtbot.mouseClick(row.value, Qt.MouseButton.LeftButton, pos=QPoint(7, row.value.height() // 2))
    loop.run_until_complete(row.write())
    assert device.current is False
    device.current = 'text'
    loop.run_until_complete(row.read())
    assert row.value.text() == 'text'
    device.current = [1, 2]
    loop.run_until_complete(row.read())
    assert not row.write_button.isEnabled()
    loop.run_until_complete(widget.shutdown())


def test_quantity_metadata_and_custom_editor(qtbot, loop):
    class CustomEditor(ScalarEditor):
        pass

    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device, editors={'value': CustomEditor})
    qtbot.addWidget(widget)
    assert isinstance(widget.params['value'].editor, CustomEditor)
    row = widget.params['position']
    loop.run_until_complete(row.read())
    assert row.value.text() == '2'
    assert '4' in row.target_label.text()
    assert '10' in row.unit_label.toolTip()
    row.value.setText('20')
    loop.run_until_complete(row.write())
    assert 'upper limit' in row.error_label.text()
    loop.run_until_complete(widget.shutdown())


def test_styling_and_error_visibility_are_local(qtbot, loop, qapp):
    original_style = qapp.styleSheet()
    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device)
    qtbot.addWidget(widget)
    row = widget.params['position']
    assert qapp.styleSheet() == original_style
    assert row.objectName() == 'concertParameterRow'
    assert row.value.accessibleName() == 'Value: position'
    assert row.error_label.isHidden()
    assert row.target_label.isHidden()
    loop.run_until_complete(row.read())
    assert not row.target_label.isHidden()
    row.value.setText('20')
    loop.run_until_complete(row.write())
    assert not row.error_label.isHidden()
    assert row.status_label.property('tone') == 'error'
    row.value.setText('3')
    loop.run_until_complete(row.write())
    assert row.error_label.isHidden()
    loop.run_until_complete(widget.shutdown())


def test_read_age_updates_and_write_does_not_reset_it(qtbot, loop):
    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device)
    qtbot.addWidget(widget)
    row = widget.params['value']
    assert row.status_label.text() == 'Not read yet'
    loop.run_until_complete(row.read())
    timestamp = row.controller.last_read_at
    for elapsed, expected in ((5, '5s ago'), (125, '2m ago'),
                              (7200, '2h ago'), (172800, '2d ago')):
        with patch('concert.gui.parameterizable.time.monotonic', return_value=timestamp + elapsed):
            row._update_controls()
            assert row.status_label.text() == expected
    loop.run_until_complete(row.write())
    assert row.controller.last_read_at == timestamp
    loop.run_until_complete(widget.shutdown())
    assert not widget._scheduler.timer.isActive()


def test_initial_read_and_checkbox_replacement(qtbot, loop):
    class CountingDevice(Device):
        async def __ainit__(self):
            await super().__ainit__()
            self.reads = 0

        async def _get_value(self):
            self.reads += 1
            return self.current

    device = loop.run_until_complete(CountingDevice())
    widget = ParameterizableWidget(device)
    qtbot.addWidget(widget)
    widget.show()
    row = widget.params['value']
    old_editor = row.value

    async def wait_for_initial_reads():
        for _ in range(100):
            if all(item.controller.last_read_at is not None for item in widget.params.values()):
                return
            await asyncio.sleep(0.01)
        raise AssertionError('Initial reads did not complete')

    loop.run_until_complete(wait_for_initial_reads())
    assert device.reads == 1
    assert isinstance(row.value, QCheckBox)
    assert row.value.text() == 'Enabled'
    from PyQt6 import sip
    assert sip.isdeleted(old_editor) or old_editor.isHidden()
    loop.run_until_complete(asyncio.sleep(0.1))
    assert device.reads == 1
    loop.run_until_complete(widget.shutdown())


def test_polling_pauses_and_resumes_across_tabs(qtbot, loop):
    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device)
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.addTab(widget, 'Device')
    tabs.addTab(QWidget(), 'Other')
    tabs.resize(600, 700)
    tabs.show()
    loop.run_until_complete(widget.refresh())
    row = widget.params['value']
    row.set_polling(True)
    assert row.polling_active
    tabs.setCurrentIndex(1)
    assert row.polling and row.poll_action.isChecked()
    assert not row.polling_active
    tabs.setCurrentIndex(0)
    loop.run_until_complete(asyncio.sleep(0.05))
    assert row.polling and row.poll_action.isChecked()
    assert row.polling_active
    row.set_polling(False)
    tabs.setCurrentIndex(1)
    tabs.setCurrentIndex(0)
    assert not row.polling_active
    loop.run_until_complete(widget.shutdown())


def test_only_visible_rows_are_read_and_scrolling_resumes_polling(qtbot, loop):
    class ManyParameters(Parameterizable):
        pass

    # A tall list ensures the final parameter is clipped by the viewport.
    for index in range(12):
        async def getter(instance, index=index):
            instance.reads[index] += 1
            return index
        setattr(ManyParameters, f'item_{index:02}', Parameter(fget=getter))

    device = loop.run_until_complete(ManyParameters())
    device.reads = [0] * 12
    widget = ParameterizableWidget(device)
    qtbot.addWidget(widget)
    widget.resize(400, 250)
    loop.run_until_complete(widget.refresh())
    assert device.reads == [0] * 12
    widget.show()
    loop.run_until_complete(asyncio.sleep(0.15))
    first = widget.params['item_00']
    last = widget.params['item_11']
    assert device.reads[0] == 1 and device.reads[11] == 0
    assert first.is_value_visible() and not last.is_value_visible()
    widget.set_polling(True)
    widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())
    loop.run_until_complete(asyncio.sleep(0.15))
    assert device.reads[11] >= 1
    assert not first.polling_active and last.polling_active
    assert first.polling and last.polling
    widget.hide()
    before = device.reads.copy()
    loop.run_until_complete(asyncio.sleep(0.6))
    assert device.reads == before
    loop.run_until_complete(widget.shutdown())


def test_alignment_window_close_and_nonblocking_confirmation(qtbot, loop):
    alignment = loop.run_until_complete(Alignment(
        None, None, None, None, None, 1 * q.mm,
        target_horizontal=5, target_vertical=5))
    qtbot.addWidget(alignment._viewer)
    alignment._viewer.show()
    alignment._viewer.close()
    assert alignment._input_event.is_set() and alignment._last_pos is None

    def reject():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMessageBox):
                widget.button(QMessageBox.StandardButton.No).click()

    QTimer.singleShot(10, reject)
    assert loop.run_until_complete(alignment._confirm((1 * q.mm, 2 * q.mm))) is False


def test_hiding_widget_preserves_submitted_write(qtbot, loop):
    class SlowDevice(Device):
        async def _set_value(self, value):
            started.set()
            await release.wait()
            self.current = value

    release = asyncio.Event()
    started = asyncio.Event()
    device = loop.run_until_complete(SlowDevice())
    widget = ParameterizableWidget(device)
    qtbot.addWidget(widget)
    widget.show()
    row = widget.params['value']
    loop.run_until_complete(row.read())
    row.value.setChecked(False)
    async def submit():
        row.request_write()
        await asyncio.wait_for(started.wait(), 2)

    loop.run_until_complete(submit())
    widget.hide()
    assert row.controller.busy
    release.set()
    loop.run_until_complete(widget.shutdown())
    assert device.current is False


def test_dirty_editor_retains_input_and_reset_uses_latest_observation(qtbot, loop):
    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device, initial_read=False, refresh_when_visible=False)
    qtbot.addWidget(widget)
    row = widget.params['position']
    loop.run_until_complete(row.read())
    row.value.setText('5')
    assert row.dirty and row.reset_button.isEnabled()
    loop.run_until_complete(row.read())
    assert row.value.text() == '5'
    assert '2' in row.actual_label.text() and '4' in row.target_label.text()
    assert 'Effective limits: 0 mm' in row.limits_label.text()
    assert '10 mm' in row.limits_label.text()
    assert 'User limits' in row.limits_label.text()
    assert 'External limits' in row.limits_label.text()
    row.reset_edit()
    assert not row.dirty and row.value.text() == '2'
    row.value.setText('6')
    loop.run_until_complete(row.write())
    assert not row.dirty
    loop.run_until_complete(widget.shutdown())


def test_write_only_metadata_groups_options_and_session_cleanup(qtbot, loop):
    class WriteOnly(Parameterizable):
        command = Selection(['start', 'stop'], help='A write-only command')

        async def _set_command(self, value):
            self.commanded = value

    device = loop.run_until_complete(WriteOnly())
    param = device['command']
    assert not param.readable and param.writable and param.help == 'A write-only command'
    session = GuiSession()
    widget = ParameterizableWidget(device, session=session,
                                   groups={'Commands': ['command']},
                                   parameter_options={'command': {'poll_interval': 2 * q.s}})
    qtbot.addWidget(widget)
    widget.show()
    row = widget.params['command']
    assert row.poll_interval == 2
    assert not row.read_button.isEnabled()
    row.set_polling(True)
    assert not row.polling
    row.value.setCurrentIndex(0)
    loop.run_until_complete(row.write())
    assert device.commanded == 'start'
    group = row.parentWidget()
    group.setChecked(False)
    assert not row.is_value_visible()
    group.setChecked(True)
    assert row.isVisible()
    loop.run_until_complete(session.shutdown())
    assert row.controller.closed and not widget._scheduler.timer.isActive()
    assert not session._controllers


def test_metadata_failure_is_separate_and_panel_reports_validation(qtbot, loop):
    class BrokenMetadata(Device):
        async def _get_target_position(self):
            raise RuntimeError('target offline')

    device = loop.run_until_complete(BrokenMetadata())
    widget = ParameterizableWidget(device)
    qtbot.addWidget(widget)
    failures = []
    widget.operation_failed.connect(lambda *args: failures.append(args))
    row = widget.params['position']
    loop.run_until_complete(row.read())
    assert row.value.text() == '2'
    assert row.error_label.isHidden()
    assert 'target offline' in row.metadata_label.text()
    assert failures[0][:2] == ('position', 'metadata')
    row.value.setText('not a number')
    loop.run_until_complete(row.write())
    assert failures[-1][:2] == ('position', 'validation')
    loop.run_until_complete(widget.shutdown())


def test_scheduler_is_shared_and_intervals_are_validated(qtbot, loop):
    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device, poll_interval=2 * q.s,
                                   initial_read=False, refresh_when_visible=False)
    qtbot.addWidget(widget)
    assert all(row._scheduler is widget._scheduler for row in widget.params.values())
    widget.resize(600, 700)
    widget.show()
    loop.run_until_complete(asyncio.sleep(0.1))
    assert all(row.controller.last_read_at is None for row in widget.params.values())
    widget.set_polling(True)
    loop.run_until_complete(asyncio.sleep(0.15))
    timestamps = [row.controller.last_read_at for row in widget.params.values()]
    assert all(stamp is not None for stamp in timestamps)
    loop.run_until_complete(asyncio.sleep(0.2))
    assert timestamps == [row.controller.last_read_at for row in widget.params.values()]
    loop.run_until_complete(widget.shutdown())
    from concert.gui.refresh import interval_seconds
    with pytest.raises(ValueError):
        interval_seconds(0 * q.s)
    with pytest.raises(ValueError):
        interval_seconds(float('nan') * q.s)


def test_deleted_panel_session_still_waits_for_write(qtbot, loop):
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent

    started, release = asyncio.Event(), asyncio.Event()

    class SlowDevice(Device):
        async def _set_value(self, value):
            started.set()
            await release.wait()
            self.current = value

    device = loop.run_until_complete(SlowDevice())
    session = GuiSession()
    widget = ParameterizableWidget(device, session=session)
    qtbot.addWidget(widget)
    row = widget.params['value']
    loop.run_until_complete(row.read())
    row.value.setChecked(False)

    async def exercise():
        row.request_write()
        await started.wait()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(widget)
        cleanup = asyncio.create_task(session.shutdown())
        await asyncio.sleep(0)
        assert not cleanup.done()
        release.set()
        await cleanup

    loop.run_until_complete(exercise())
    assert device.current is False and not session._controllers


def test_checkbox_and_selection_edits_survive_refresh(qtbot, loop):
    class Choices(Device):
        mode = Selection(['a', 'b'])

        async def _get_mode(self):
            return 'a'

        async def _set_mode(self, value):
            pass

    device = loop.run_until_complete(Choices())
    widget = ParameterizableWidget(device, initial_read=False)
    qtbot.addWidget(widget)
    flag, mode = widget.params['value'], widget.params['mode']
    loop.run_until_complete(flag.read())
    loop.run_until_complete(mode.read())
    flag.value.setChecked(False)
    mode.value.setCurrentIndex(1)
    loop.run_until_complete(flag.read())
    loop.run_until_complete(mode.read())
    assert flag.dirty and not flag.value.isChecked()
    assert mode.dirty and mode.value.currentIndex() == 1
    flag.reset_edit()
    mode.reset_edit()
    assert flag.value.isChecked() and mode.value.currentIndex() == 0
    loop.run_until_complete(widget.shutdown())


def test_limits_are_collapsed_editable_and_locked(qtbot, loop):
    async def apply(row, name):
        row.request_limit(name)
        await row.controller._task

    device = loop.run_until_complete(Device())
    widget = ParameterizableWidget(device, initial_read=False)
    qtbot.addWidget(widget)
    row = widget.params['position']
    loop.run_until_complete(row.read())
    assert row.limits_panel.isHidden() and not row.limits_toggle.isChecked()
    assert row.limit_editors['lower'].widget.text() == '0'
    row.limits_toggle.setChecked(True)
    assert not row.limits_panel.isHidden()
    timestamp = row.controller.last_read_at
    row.limit_editors['upper'].widget.setText('8 mm')
    loop.run_until_complete(apply(row, 'upper'))
    assert loop.run_until_complete(device['position'].get_upper_user()) == 8 * q.mm
    assert row.controller.last_read_at == timestamp
    assert row.limit_editors['upper'].widget.text() == '8'
    assert '8 mm' in row.limits_label.text()
    # Refreshes do not overwrite unsaved limit edits.
    row.limit_editors['lower'].widget.setText('3')
    loop.run_until_complete(row.read())
    assert row.limit_editors['lower'].widget.text() == '3'
    row.reset_limits()
    assert row.limit_editors['lower'].widget.text() == '0'
    # Reject crossed bounds; leave the existing bound in place.
    row.limit_editors['lower'].widget.setText('9')
    loop.run_until_complete(apply(row, 'lower'))
    assert row.controller.error is not None
    assert loop.run_until_complete(device['position'].get_lower_user()) == 0 * q.mm
    row.limit_editors['upper'].widget.clear()
    loop.run_until_complete(apply(row, 'upper'))
    assert loop.run_until_complete(device['position'].get_upper_user()) is None
    device['position'].lock_limits()
    row._update_controls()
    assert not row.limit_buttons['lower'].isEnabled()
    # Also guard programmatic requests, including clearing a locked bound.
    row.limit_editors['lower'].widget.clear()
    loop.run_until_complete(apply(row, 'lower'))
    assert 'locked' in str(row.controller.error)
    assert loop.run_until_complete(device['position'].get_lower_user()) == 0 * q.mm
    row.limits_toggle.setChecked(False)
    assert row.limits_panel.isHidden()
    loop.run_until_complete(widget.shutdown())
