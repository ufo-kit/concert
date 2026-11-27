"""Qt presentation for the independent parameter controller and editor adapters."""
import asyncio
import time

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QTimer, Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QLabel, QPushButton, QHBoxLayout,
                             QVBoxLayout, QStyle, QMenu, QScrollArea, QSizePolicy,
                             QGroupBox, QLineEdit, QCheckBox, QComboBox, QToolButton)

from concert.base import Parameterizable, QuantityValue, StateValue, SelectionValue
from concert.gui.controller import ParameterController, UNREAD
from concert.gui.editors import EditorRegistry, QuantityEditor
from concert.gui.style import apply_style, set_status
from concert.gui.refresh import RefreshScheduler, PanelLifecycle, interval_seconds
from concert.quantities import q


class ParameterizableWidget(QScrollArea):
    """Present parameters with optional grouping and quantity-based refresh settings.

    ``parameter_options`` overrides row settings by name. ``groups`` maps titles
    to parameter names; ungrouped parameters follow ``order`` then device order.
    ``operation_failed`` emits (parameter name, read/write/limits/validation/metadata, error).
    """
    operation_failed = pyqtSignal(str, str, Exception)

    def __init__(self, parameterizable, exclude_properties=None, parent=None,
                 editors=None, registry=None, session=None, *, poll_interval=0.5 * q.s,
                 initial_read=True, refresh_when_visible=True, parameter_options=None,
                 groups=None, order=None):
        if not isinstance(parameterizable, Parameterizable):
            raise TypeError('Only Parameterizables can be wrapped')
        super().__init__(parent)
        self.setObjectName('concertParameters')
        self.setMinimumWidth(320)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName('concertContent')
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(10)
        self.setWidget(content)
        apply_style(self)
        self.params = {}
        self._detach_session = None
        self._scheduler = RefreshScheduler(self)
        registry = registry or EditorRegistry()
        parameters = {param.name: param for param in parameterizable
                      if param.name not in (exclude_properties or ())}
        groups = groups or {}
        names = [name for members in groups.values() for name in members]
        names += list(order or ())
        unknown = set(names) - set(parameters)
        if unknown:
            raise ValueError(f'Unknown or excluded parameters: {sorted(unknown)}')
        grouped = [name for members in groups.values() for name in members]
        if len(grouped) != len(set(grouped)):
            raise ValueError('A parameter may belong to only one group')
        names = list(dict.fromkeys(names + list(parameters)))
        group_layouts = {}
        for title in groups:
            group = QGroupBox(title, content)
            group.setObjectName('concertGroup')
            group.setCheckable(True)
            group.setChecked(True)
            group_layouts[title] = QVBoxLayout(group)
            self._layout.addWidget(group)
            group.toggled.connect(lambda checked, group=group: self._toggle_group(group, checked))
        for name in names:
            param = parameters[name]
            widget_type = (SelectionWidget if isinstance(param, SelectionValue) else
                           QuantityWidget if isinstance(param, QuantityValue) else
                           StateWidget if isinstance(param, StateValue) else ParameterWidget)
            options = dict(poll_interval=poll_interval, initial_read=initial_read,
                           refresh_when_visible=refresh_when_visible)
            options.update((parameter_options or {}).get(name, {}))
            row = widget_type(param, content, registry.create(param, (editors or {}).get(name)),
                              scheduler=self._scheduler, **options)
            self.params[param.name] = row
            row.destroyed.connect(lambda _obj=None, name=param.name: self.params.pop(name, None))
            row.operation_failed.connect(self.operation_failed.emit)
            layout = next((group_layouts[title] for title, members in groups.items()
                           if name in members), self._layout)
            layout.addWidget(row)
        self._layout.addStretch()
        self.verticalScrollBar().valueChanged.connect(self._update_visibility)
        self.horizontalScrollBar().valueChanged.connect(self._update_visibility)
        if session is not None:
            self._detach_session = session.register(PanelLifecycle(self))

    def _toggle_group(self, group, checked):
        for index in range(group.layout().count()):
            group.layout().itemAt(index).widget().setVisible(checked)
        self._update_visibility()

    def add_widget(self, widget):
        self._layout.insertWidget(self._layout.count() - 1, widget)

    async def refresh(self):
        await asyncio.gather(*(row.read() for row in tuple(self.params.values())
                                if row.is_value_visible() and row._param.readable))

    def _update_visibility(self):
        for row in tuple(self.params.values()):
            row.update_visibility()

    def set_polling(self, enabled):
        for row in self.params.values():
            row.set_polling(enabled)

    def closeEvent(self, event):
        self._scheduler.stop()
        for row in self.params.values():
            row.stop()
        super().closeEvent(event)

    async def shutdown(self):
        self._scheduler.stop()
        await asyncio.gather(*(row.shutdown() for row in tuple(self.params.values())))
        if self._detach_session is not None:
            self._detach_session()
            self._detach_session = None

    def event(self, event):
        if event.type() == QEvent.Type.DeferredDelete and hasattr(self, '_scheduler'):
            self._scheduler.stop()
            for row in tuple(self.params.values()):
                row.stop()
            if self._detach_session is not None:
                # Deleted rows still own submitted writes. Detach only after
                # those operations finish, or let the session wait for them.
                controllers = [row.controller for row in self.params.values()]
                detach = self._detach_session
                async def finish():
                    await asyncio.gather(*(controller.shutdown() for controller in controllers))
                    detach()
                try:
                    asyncio.get_running_loop().create_task(finish())
                except RuntimeError:
                    pass
                self._detach_session = None
        return super().event(event)


class ParameterWidget(QWidget):
    operation_failed = pyqtSignal(str, str, Exception)
    setter_started = pyqtSignal()
    setter_finished = pyqtSignal()
    setter_error = pyqtSignal(Exception)
    getter_started = pyqtSignal()
    getter_finished = pyqtSignal()
    getter_error = pyqtSignal(Exception)

    def __init__(self, param, parent=None, editor=None, *, scheduler=None,
                 poll_interval=0.5 * q.s, initial_read=True, refresh_when_visible=True):
        super().__init__(parent)
        self.setObjectName('concertParameterRow')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        apply_style(self)
        self.controller = ParameterController(param)
        self.editor = editor or EditorRegistry().create(param)
        self._param = param
        self.polling = False
        self.poll_interval = interval_seconds(poll_interval)
        self.refresh_when_visible = refresh_when_visible
        self.dirty = False
        self._displaying = False
        self._next_poll = 0
        self._explicit_read = 0
        self.name_label = QLabel(param.name.replace('_', ' ').capitalize())
        self.name_label.setObjectName('concertName')
        self.name_label.setAccessibleName(param.name)
        self.name_label.setToolTip(self.controller.help)
        self.value = self.editor.widget
        self.value.setToolTip(self.controller.help)
        self.status_label = QLabel('Not read yet')
        self.status_label.setObjectName('concertStatus')
        self.target_label = QLabel()
        self.target_label.setObjectName('concertTarget')
        self.target_label.setWordWrap(True)
        self.target_label.hide()
        self.actual_label = QLabel()
        self.actual_label.setObjectName('concertTarget')
        self.actual_label.hide()
        self.limits_label = QLabel()
        self.limits_label.setObjectName('concertTarget')
        self.limits_label.setWordWrap(True)
        self.limits_label.hide()
        self._limit_dirty = set()
        self.limit_editors = {}
        self.limit_buttons = {}
        self.limits_toggle = QToolButton()
        self.limits_toggle.setText('Limits')
        self.limits_toggle.setCheckable(True)
        self.limits_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.limits_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.limits_toggle.setAccessibleName(f'Expand limits: {param.name}')
        self.limits_panel = QWidget()
        limits_layout = QVBoxLayout(self.limits_panel)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        limits_layout.addWidget(self.limits_label)
        if isinstance(param, QuantityValue):
            for name in ('lower', 'upper'):
                editor = QuantityEditor(param)
                # Limits can be edited even for a read-only value.
                if not isinstance(editor.widget, QLineEdit):
                    editor.widget = QLineEdit()
                editor.widget.setObjectName('concertEditor')
                editor.widget.setPlaceholderText('Unbounded (empty)')
                editor.widget.setAccessibleName(f'User {name} limit: {param.name}')
                editor.widget.textChanged.connect(lambda _text, name=name: self._limit_dirty.add(name))
                button = QPushButton('Apply')
                button.setObjectName('concertAction')
                button.setAccessibleName(f'Apply {name} limit: {param.name}')
                button.clicked.connect(lambda _checked=False, name=name: self.request_limit(name))
                line = QHBoxLayout()
                line.addWidget(QLabel(f'User {name}'))
                line.addWidget(editor.widget, 1)
                line.addWidget(QLabel(format(param.unit, '~')))
                line.addWidget(button)
                limits_layout.addLayout(line)
                self.limit_editors[name] = editor
                self.limit_buttons[name] = button
            note = QLabel('Empty removes the user bound. External limits remain enforced.')
            note.setObjectName('concertTarget')
            note.setWordWrap(True)
            limits_layout.addWidget(note)
            self.limits_reset = QPushButton('Reset limit edits')
            self.limits_reset.setObjectName('concertAction')
            self.limits_reset.clicked.connect(self.reset_limits)
            limits_layout.addWidget(self.limits_reset)
        self.limits_panel.hide()
        self.limits_toggle.setVisible(isinstance(param, QuantityValue))
        self.limits_toggle.toggled.connect(self._toggle_limits)
        self.metadata_label = QLabel()
        self.metadata_label.setObjectName('concertTarget')
        self.metadata_label.setWordWrap(True)
        self.metadata_label.hide()
        self.unit_label = QLabel(format(param.unit, '~') if isinstance(param, QuantityValue) else '')
        self.unit_label.setObjectName('concertUnit')
        self.unit_label.setVisible(bool(self.unit_label.text()))
        self.error_label = QLabel()
        self.error_label.setObjectName('concertError')
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.read_button = QPushButton()
        self.read_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.read_button.setToolTip('Read actual value')
        self.write_button = QPushButton()
        self.write_button.setText('Apply')
        self.reset_button = QPushButton('Reset')
        self.reset_button.setObjectName('concertAction')
        self.reset_button.setAccessibleName(f'Reset edit: {param.name}')
        self.reset_button.clicked.connect(self.reset_edit)
        self.write_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.read_button.clicked.connect(self.request_refresh)
        self.write_button.clicked.connect(self.request_write)
        for button, name in ((self.read_button, 'Refresh'), (self.write_button, 'Apply value')):
            button.setObjectName('concertAction')
            button.setAccessibleName(f'{name}: {param.name}')
            button.setIconSize(QSize(16, 16))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.read_button.setFixedSize(36, 36)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(8)
        heading = QHBoxLayout()
        heading.addWidget(self.name_label, 1)
        heading.addWidget(self.status_label)
        self._layout.addLayout(heading)
        self._editor_layout = QHBoxLayout()
        self._editor_layout.setSpacing(8)
        self._editor_layout.addWidget(self.value, 1)
        for widget in (self.unit_label, self.read_button, self.write_button, self.reset_button):
            self._editor_layout.addWidget(widget)
        self._layout.addLayout(self._editor_layout)
        self._layout.addWidget(self.target_label)
        self._layout.addWidget(self.actual_label)
        self._layout.addWidget(self.limits_toggle)
        self._layout.addWidget(self.limits_panel)
        self._layout.addWidget(self.metadata_label)
        self._layout.addWidget(self.error_label)
        self._prepare_editor()
        self.context_menu = QMenu(self)
        self.poll_action = self.context_menu.addAction('Poll')
        self.poll_action.setCheckable(True)
        self.poll_action.toggled.connect(self.set_polling)
        self.context_menu.addAction('Hide').triggered.connect(self.hide)
        self._visible = False
        self._initial_read_enabled = initial_read and param.readable
        self._initial_read_pending = initial_read and param.readable
        self._owns_scheduler = scheduler is None
        self._scheduler = scheduler or RefreshScheduler(self)
        self._scheduler.rows.append(self)
        self.controller.subscribe(self._changed)
        self.destroyed.connect(self._on_destroyed)
        self._update_controls()

    def _initial_read(self):
        if self.controller.closed:
            return
        if self.controller.last_read_at is not None:
            self._initial_read_pending = False
            return
        if self.controller.busy:
            return
        if not self.is_value_visible():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._initial_read_pending = False
        self.request_refresh()

    def is_value_visible(self):
        """True only when this row occupies an unclipped area of a visible window."""
        return (not self.controller.closed and self.isVisible()
                and not self.window().isMinimized() and not self.visibleRegion().isEmpty())

    def update_visibility(self):
        visible = self.is_value_visible()
        became_visible = visible and not self._visible
        self._visible = visible
        if not visible:
            if not self._explicit_read:
                self.controller.cancel_read()
            return
        if self._initial_read_pending:
            self._initial_read()
        now = time.monotonic()
        if self.polling_active and (now >= self._next_poll or became_visible):
            self.request_refresh()
            self._next_poll = now + self.poll_interval
        elif became_visible and self.refresh_when_visible and self.controller.last_read_at is not None:
            self.request_refresh()

    @property
    def polling_active(self):
        return self.polling and self._param.readable and self.is_value_visible()

    def _prepare_editor(self):
        self.value.setObjectName('concertEditor')
        self.value.setMinimumWidth(60)
        self.value.setMinimumHeight(34)
        self.value.setAccessibleName(f'Value: {self._param.name}')
        self.value.setToolTip(self.controller.help)
        if hasattr(self.value, 'setPlaceholderText'):
            self.value.setPlaceholderText('Read to load value')
        if isinstance(self.value, QLineEdit):
            self.value.textChanged.connect(self._edited)
        elif isinstance(self.value, QCheckBox):
            self.value.toggled.connect(self._edited)
        elif isinstance(self.value, QComboBox):
            self.value.currentIndexChanged.connect(self._edited)

    def _edited(self, *args):
        if not self._displaying:
            self.mark_dirty()

    def mark_dirty(self):
        """Custom editors call this when the operator changes their input."""
        self.dirty = True
        self._update_controls()

    def _display_value(self, value):
        self._displaying = True
        try:
            old = self.value
            self.editor.display(value)
            self.value = self.editor.widget
            if self.value is not old:
                self._editor_layout.replaceWidget(old, self.value)
                old.hide()
                old.deleteLater()
                self._prepare_editor()
        finally:
            self._displaying = False

    def reset_edit(self):
        if self.controller.value is not UNREAD:
            self._display_value(self.controller.value)
            self.dirty = False
            self._update_controls()

    def _toggle_limits(self, expanded):
        self.limits_panel.setVisible(expanded)
        self.limits_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def reset_limits(self):
        self._limit_dirty.clear()
        self._display_limits()

    def _display_limits(self):
        for name, editor in self.limit_editors.items():
            if name in self._limit_dirty:
                continue
            value = self.controller.limits.get(name + '_user', UNREAD)
            blocked = editor.widget.blockSignals(True)
            try:
                editor.widget.setText('' if value is UNREAD or value is None else
                                      str(value.to(self._param.unit).magnitude))
            finally:
                editor.widget.blockSignals(blocked)

    def request_limit(self, name):
        if self.controller.busy or self.controller.closed:
            return
        try:
            editor = self.limit_editors[name]
            value = editor.parse() if editor.widget.text().strip() else None
            self.controller.request('limits', (name, value))
            self._limit_dirty.add(name)
        except (ValueError, TypeError, RuntimeError) as error:
            self._show_error(str(error))
            self.operation_failed.emit(self._param.name, 'validation', error)

    def _show_error(self, message):
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
        if message:
            set_status(self.status_label, 'Error', 'error')

    def _update_controls(self):
        controller = self.controller
        enabled = not (controller.busy or controller.closed)
        locked = self._param.locked
        editable = enabled and not locked and self.editor.editable
        self.read_button.setEnabled(enabled and self._param.readable)
        self.write_button.setEnabled(editable)
        # Keep input/focus usable during automatic reads; Apply remains serialized.
        self.value.setEnabled(not controller.closed and not locked and self.editor.editable
                              and not (controller.busy and controller.operation == 'write'))
        self.reset_button.setEnabled(enabled and self.dirty and controller.value is not UNREAD)
        self.reset_button.setVisible(self._param.writable)
        if self.limit_editors:
            limits_enabled = enabled and not self._param.limits_locked
            for name, editor in self.limit_editors.items():
                editor.widget.setEnabled(limits_enabled)
                self.limit_buttons[name].setEnabled(limits_enabled)
                self.limit_buttons[name].setToolTip('Limits are locked' if self._param.limits_locked
                                                     else 'Set user limit (does not move the device)')
        self.write_button.setVisible(self._param.writable)
        reason = ('Closed' if controller.closed else 'Operation pending' if controller.busy else
                  'Parameter is locked' if locked else
                  'Read-only or unsupported editor' if not self.editor.editable else 'Write value')
        self.write_button.setToolTip(reason)
        if not controller.busy and not self.error_label.text():
            text = ('Closed' if controller.closed else 'Locked' if locked else
                    'Write-only' if not self._param.readable else
                    'Not read yet' if controller.last_read_at is None else self._read_age())
            if self.dirty:
                text = 'Edited · ' + text
            set_status(self.status_label, text, 'active' if self.polling else 'neutral')

    def _read_age(self):
        seconds = max(0, int(time.monotonic() - self.controller.last_read_at))
        if seconds < 60:
            return f'{seconds}s ago'
        if seconds < 3600:
            return f'{seconds // 60}m ago'
        if seconds < 86400:
            return f'{seconds // 3600}h ago'
        return f'{seconds // 86400}d ago'

    def _changed(self, event, controller):
        if sip.isdeleted(self):
            return
        writing = controller.operation == 'write'
        if event == 'started':
            text = ('Setting limit…' if controller.operation == 'limits' else
                    'Writing…' if writing else 'Reading…')
            set_status(self.status_label, text, 'active')
            if controller.operation != 'limits':
                (self.setter_started if writing else self.getter_started).emit()
        elif event in ('value', 'limits'):
            if event == 'limits':
                self._limit_dirty.discard(controller.updated_limit)
            if event == 'value' and not self.dirty:
                self._display_value(controller.value)
            if controller.value is not UNREAD:
                self.actual_label.setText(f'Observed: {controller.value}')
                self.actual_label.setVisible(self._param.writable)
            self.target_label.setText('' if controller.target is UNREAD else f'Target: {controller.target}')
            self.target_label.setVisible(controller.target is not UNREAD)
            self.unit_label.setToolTip(f'Limits: {controller.lower} … {controller.upper}')
            if isinstance(self._param, QuantityValue):
                lines = []
                for title, suffix in (('Effective limits', ''), ('User limits', '_user'),
                                      ('External limits', '_external')):
                    lower = controller.limits.get('lower' + suffix, UNREAD)
                    upper = controller.limits.get('upper' + suffix, UNREAD)
                    def bound(value):
                        return ('Unavailable' if value is UNREAD else 'Unbounded' if value is None
                                else format(value, '~'))
                    lines.append(f'{title}: {bound(lower)} … {bound(upper)}')
                lines.append('Limits locked' if self._param.limits_locked else 'Limits unlocked')
                self.limits_label.setText('\n'.join(lines))
                self.limits_label.show()
                self._display_limits()
            self.metadata_label.setText('' if controller.metadata_error is None else
                                        f'Metadata unavailable: {controller.metadata_error}')
            self.metadata_label.setVisible(controller.metadata_error is not None)
        elif event == 'written':
            # Do not clear an edit made after submission (custom adapters may allow it).
            try:
                if self.editor.parse() == controller.written_value:
                    self.dirty = False
            except (ValueError, TypeError):
                pass
        elif event == 'metadata_error':
            self.operation_failed.emit(self._param.name, 'metadata', controller.metadata_error)
        elif event == 'error':
            if controller.operation != 'limits':
                (self.setter_error if writing else self.getter_error).emit(controller.error)
            self.operation_failed.emit(self._param.name, controller.operation, controller.error)
        elif event == 'finished':
            if controller.last_read_at is None and self._initial_read_enabled and controller.error is None:
                self._initial_read_pending = True
            self._show_error(str(controller.error or ''))
            if controller.error:
                self.status_label.setToolTip('Last successfully read value is retained')
            if controller.operation != 'limits':
                (self.setter_finished if writing else self.getter_finished).emit()
        self._update_controls()

    def set_polling(self, enabled):
        self.polling = bool(enabled) and not self.controller.closed and self._param.readable
        self.poll_action.setChecked(self.polling)
        self._next_poll = 0
        self._update_controls()

    def contextMenuEvent(self, event):
        self.context_menu.exec(self.mapToGlobal(event.pos()))

    def request_refresh(self):
        if not self.is_value_visible() or not self._param.readable:
            return
        try:
            asyncio.get_running_loop()
            self.controller.request('read')
            self._update_controls()
        except RuntimeError:
            # Synchronous IPython construction has no running loop yet.
            return

    def request_write(self):
        if self.controller.busy or self.controller.closed:
            return
        try:
            self.controller.request('write', self.editor.parse())
            self._update_controls()
        except (ValueError, TypeError, RuntimeError) as error:
            self._show_error(str(error))
            self.setter_error.emit(error)
            self.operation_failed.emit(self._param.name, 'validation', error)

    async def read(self):
        self._explicit_read += 1
        try:
            await self.controller.read()
        finally:
            self._explicit_read -= 1
        if not sip.isdeleted(self):
            self._update_controls()

    async def write(self):
        if self.controller.busy or self.controller.closed:
            return
        try:
            await self.controller.write(self.editor.parse())
        except (ValueError, TypeError) as error:
            if not sip.isdeleted(self):
                self._show_error(str(error))
                self.setter_error.emit(error)
                self.operation_failed.emit(self._param.name, 'validation', error)
        finally:
            if not sip.isdeleted(self):
                self._update_controls()

    def stop(self):
        self.polling = False
        if self._owns_scheduler:
            self._scheduler.stop()
        if self in self._scheduler.rows:
            self._scheduler.rows.remove(self)
        self.controller.close()

    def _on_destroyed(self):
        if self in self._scheduler.rows:
            self._scheduler.rows.remove(self)
        self.controller.unsubscribe(self._changed)
        self.controller.close()

    async def shutdown(self):
        self.stop()
        await self.controller.shutdown()

    def hideEvent(self, event):
        # Pause work while offscreen without losing the user's Poll preference.
        self._visible = False
        if not self._explicit_read:
            self.controller.cancel_read()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.update_visibility)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.DeferredDelete:
            self.stop()
        return super().event(event)


class StateWidget(ParameterWidget):
    pass


class QuantityWidget(ParameterWidget):
    pass


class SelectionWidget(ParameterWidget):
    pass
