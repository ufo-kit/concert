"""Async editors for Concert parameters; constructors never access hardware."""
import asyncio

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout,
                             QVBoxLayout, QLineEdit, QCheckBox, QStyle, QMenu, QScrollArea)

from concert.base import Parameterizable, QuantityValue, StateValue, SelectionValue
from concert.quantities import q


class ParameterizableWidget(QScrollArea):
    def __init__(self, parameterizable, exclude_properties=None, parent=None):
        if not isinstance(parameterizable, Parameterizable):
            raise TypeError('Only Parameterizables can be wrapped')
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        content = QWidget()
        self._layout = QVBoxLayout(content)
        self.setWidget(content)
        self.params = {}
        for param in parameterizable:
            if param.name in (exclude_properties or ()):
                continue
            widget_type = (SelectionWidget if isinstance(param, SelectionValue) else
                           QuantityWidget if isinstance(param, QuantityValue) else
                           StateWidget if isinstance(param, StateValue) else ParameterWidget)
            row = widget_type(param, content)
            self.params[param.name] = row
            row.destroyed.connect(lambda _obj=None, name=param.name: self.params.pop(name, None))
            self._layout.addWidget(row)
        self._layout.addStretch()

    def add_widget(self, widget):
        """Append an application-specific control before the trailing stretch."""
        self._layout.insertWidget(self._layout.count() - 1, widget)

    async def refresh(self):
        await asyncio.gather(*(row.read() for row in self.params.values()))

    def set_polling(self, enabled):
        for row in self.params.values():
            row.set_polling(enabled)

    def closeEvent(self, event):
        for row in self.params.values():
            row.stop()
        super().closeEvent(event)

    async def shutdown(self):
        await asyncio.gather(*(row.shutdown() for row in self.params.values()))


class ParameterWidget(QWidget):
    setter_started = pyqtSignal()
    setter_finished = pyqtSignal()
    setter_error = pyqtSignal(Exception)
    getter_started = pyqtSignal()
    getter_finished = pyqtSignal()
    getter_error = pyqtSignal(Exception)

    def __init__(self, param, parent=None):
        super().__init__(parent)
        self._param = param
        self._data_type = None
        self._busy = False
        self._stopped = False
        self._task = None
        self.polling = False
        self.name_label = QLabel(param.name)
        self.value = self.make_editor()
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.read_button = QPushButton()
        self.read_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.write_button = QPushButton()
        self.write_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.read_button.clicked.connect(lambda: self.request_refresh())
        self.write_button.clicked.connect(lambda: self.request_write())
        self._layout = QHBoxLayout(self)
        for widget in (self.name_label, self.value, self.read_button, self.write_button, self.error_label):
            self._layout.addWidget(widget)
        self.context_menu = QMenu(self)
        self.poll_action = self.context_menu.addAction('Poll')
        self.poll_action.setCheckable(True)
        self.poll_action.toggled.connect(self.set_polling)
        self.context_menu.addAction('Hide').triggered.connect(self.hide)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.request_refresh)
        self.destroyed.connect(self._on_destroyed)
        self._update_controls()

    def make_editor(self):
        return QLineEdit() if self._param.writable else QLabel()

    def can_write(self):
        return self._param.writable and self._data_type in (str, int, float, bool)

    def _update_controls(self):
        enabled = not (self._busy or self._stopped)
        self.read_button.setEnabled(enabled and not self.polling)
        self.write_button.setEnabled(enabled and not self.polling and self.can_write())
        self.value.setEnabled(enabled and not self.polling and self.can_write())

    def set_polling(self, enabled):
        self.polling = bool(enabled)
        self.poll_action.setChecked(self.polling)
        if self.polling and not self._stopped:
            self._timer.start()
        else:
            self._timer.stop()
        self._update_controls()

    def contextMenuEvent(self, event):
        self.context_menu.exec(self.mapToGlobal(event.pos()))

    def _request(self, operation):
        if self._stopped or self._busy or (self._task is not None and not self._task.done()):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.error_label.setText('Refresh requires a running asyncio event loop')
            return
        self._task = loop.create_task(operation())

    def request_refresh(self):
        self._request(self.read)

    def request_write(self):
        self._request(self.write)

    def display_value(self, value):
        self._data_type = type(value)
        if self._param.writable and isinstance(value, bool) and not isinstance(self.value, QCheckBox):
            old = self.value
            self.value = QCheckBox()
            self._layout.replaceWidget(old, self.value)
            old.deleteLater()
        if isinstance(self.value, QCheckBox):
            self.value.setChecked(bool(value))
        else:
            self.value.setText(str(value))

    def parse_value(self):
        if not self.can_write():
            raise ValueError('Read a supported scalar value before writing this parameter')
        if self._data_type is bool:
            return self.value.isChecked()
        return self._data_type(self.value.text())

    async def _operate(self, writing):
        if self._busy or self._stopped:
            return
        self._busy = True
        self._task = asyncio.current_task()
        started = self.setter_started if writing else self.getter_started
        finished = self.setter_finished if writing else self.getter_finished
        error_signal = self.setter_error if writing else self.getter_error
        self._update_controls()
        started.emit()
        try:
            if writing:
                await self._param.set(self.parse_value())
            else:
                value = await self._param.get()
                if (not self._stopped and not sip.isdeleted(self)
                        and not asyncio.current_task().cancelling()):
                    self.display_value(value)
            if not sip.isdeleted(self):
                self.error_label.clear()
        except Exception as error:
            if not sip.isdeleted(self):
                self.error_label.setText(str(error))
                error_signal.emit(error)
        finally:
            self._busy = False
            self._task = None
            if not sip.isdeleted(self):
                self._update_controls()
                finished.emit()

    async def read(self):
        await self._operate(False)

    async def write(self):
        await self._operate(True)

    def stop(self):
        self._stopped = True
        self.polling = False
        self._timer.stop()
        if self._task is not None:
            self._task.cancel()

    def _on_destroyed(self):
        # Children may already be destroyed when the parent emits this signal.
        # Only touch Python state here, never the timer or other Qt objects.
        self._stopped = True
        if self._task is not None:
            self._task.cancel()

    async def shutdown(self):
        task = self._task
        self.stop()
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)

    def hideEvent(self, event):
        self.set_polling(False)
        if self._task is not None:
            self._task.cancel()
        super().hideEvent(event)

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
    def can_write(self):
        return self._param.writable

    def parse_value(self):
        return q(self.value.text())


class SelectionWidget(ParameterWidget):
    def make_editor(self):
        editor = QComboBox()
        for value in self._param.values:
            editor.addItem(str(value), value)
        editor.setCurrentIndex(-1)
        return editor

    def can_write(self):
        return self._param.writable

    def display_value(self, value):
        index = next((i for i in range(self.value.count())
                      if self.value.itemData(i) == value), -1)
        self.value.setCurrentIndex(index)

    def parse_value(self):
        if self.value.currentIndex() < 0:
            raise ValueError('Choose a valid selection before writing')
        return self.value.currentData()
