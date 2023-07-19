from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout, QLineEdit, \
    QCheckBox, QDoubleSpinBox, QStyle, QMenu, QScrollArea
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from concert.base import Parameterizable, ParameterValue, QuantityValue, StateValue, SelectionValue
import concert.base
from concert.coroutines.base import run_in_loop
from concert.quantities import q
from qasync import asyncSlot


class ParameterizableWidget(QScrollArea):
    def __init__(self, parameterizable, exclude_properties=None):
        if exclude_properties is None:
            exclude_properties = []
        if not isinstance(parameterizable, concert.base.Parameterizable):
            raise Exception("Only Parameterizables can be wrapped.")

        self._parameterizable = parameterizable
        super().__init__()

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._exclude_properties = exclude_properties
        self.params = {}
        self.build_layout()
        self._layout.insertStretch(-1, 1)

    def build_layout(self):
        for param in self._parameterizable:
            if param.name in self._exclude_properties:
                continue
            if isinstance(param, SelectionValue):
                self.add_selection_to_layout(param)
            elif isinstance(param, QuantityValue):
                self.add_quantity_to_layout(param)
            elif isinstance(param, StateValue):
                self.add_state_to_layout(param)
            elif isinstance(param, ParameterValue):
                self.add_parameter_to_layout(param)
            else:
                raise Exception("Unknown parameter type")

    def add_state_to_layout(self, param):
        widget = StateWidget(param)
        self.params[param.name] = widget
        self._layout.addWidget(widget)

    def add_quantity_to_layout(self, param):
        widget = QuantityWidget(param)
        self.params[param.name] = widget
        self._layout.addWidget(widget)

    def add_parameter_to_layout(self, param):
        widget = ParameterWidget(param)
        self.params[param.name] = widget
        self._layout.addWidget(widget)

    def add_selection_to_layout(self, param):
        widget = SelectionWidget(param)
        self.params[param.name] = widget
        self._layout.addWidget(widget)

    def update(self):
        for param in self.params.values():
            param.update()

    def polling(self, state):
        for param in self.params.values():
            if state:
                param.deactivate()
            else:
                param.activate()


class ParameterWidget(QWidget):
    setter_started = pyqtSignal()
    setter_finished = pyqtSignal()
    setter_error = pyqtSignal(Exception)

    getter_started = pyqtSignal()
    getter_finished = pyqtSignal()
    getter_error = pyqtSignal(Exception)

    def __init__(self, param):
        super().__init__()
        self.polling = False
        self._param = param
        self.name_label = QLabel(param.name)
        self._timer = QTimer()
        self._timer.setSingleShot(False)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.update)
        self._timer.start()

        self.read_button = QPushButton()
        self.read_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.write_button = QPushButton()
        self.write_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))

        self.read_button.clicked.connect(self.read)
        self.write_button.clicked.connect(self.write)

        self.context_menu = QMenu(self)
        poll_action = self.context_menu.addAction("Poll")
        poll_action.setCheckable(True)
        poll_action.toggled.connect(self.toggle_polling)
        hide_action = self.context_menu.addAction("Hide")
        hide_action.triggered.connect(self.deleteLater)

        # TODO: in standalone the run_in_loop is not working -> fix
        try:
            self._data_type = type(run_in_loop(self._param.get()))
        except:
            self._data_type = None

        if not self._param.writable:
            self.write_button.setEnabled(False)
        if not hasattr(self, "value") and self._param.writable:
            self.value = QLineEdit()
        if not hasattr(self, "value") and not self._param.writable:
            self.value = QLabel()

        self._layout = QHBoxLayout()
        self._layout.addWidget(self.name_label)
        self._layout.addWidget(self.value)
        self._layout.addWidget(self.read_button)
        self._layout.addWidget(self.write_button)
        self.setLayout(self._layout)
        self.show()

    def toggle_polling(self):
        self.polling = not self.polling
        if self.polling:
            self.deactivate()
        else:
            self.activate()

    def contextMenuEvent(self, event):
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))

    def deactivate(self):
        self.value.setEnabled(False)
        self.write_button.setEnabled(False)
        self.read_button.setEnabled(False)

    def activate(self):
        self.read_button.setEnabled(True)
        if self._param.writable:
            self.write_button.setEnabled(True)
            self.value.setEnabled(True)

    @asyncSlot()
    async def read(self):
        if self._data_type is None:
            self._data_type = type(await self._param.get())
        self.getter_started.emit()
        try:
            self.value.setText(str(await self._param.get()))
        except Exception as e:
            self.getter_error.emit(e)
        finally:
            self.getter_finished.emit()

    @asyncSlot()
    async def write(self):
        """ Fancy parameter types are challenging. So far at the creation the type of the current value is read and
            in write is tried to cast the string to this type. A general conversion-function from str to parameter
            in the Parameter-Class would be a solution?
        """
        self.setter_started.emit()
        try:
            await self._param.set(self._data_type(self.value.text()))
        except Exception as e:
            self.setter_error.emit(e)
        finally:
            self.setter_finished.emit()

    def update(self) -> None:
        if self.polling:
            self.read()

class StateWidget(ParameterWidget):
    pass


class QuantityWidget(ParameterWidget):
    @asyncSlot()
    async def write(self):
        self.setter_started.emit()
        try:
            await self._param.set(q(self.value.text()))
        except Exception as e:
            self.setter_error.emit(e)
        finally:
            self.setter_finished.emit()


class SelectionWidget(ParameterWidget):
    def __init__(self, param):
        self.value = QComboBox()
        for value in param.values:
            self.value.addItem(str(value))
        super().__init__(param)

    @asyncSlot()
    async def read(self):
        self.getter_started.emit()
        try:
            if await self._param.get() is None:
                self.value.setCurrentText("")
            self.value.setCurrentText(str(await self._param.get()))
        except Exception as e:
            self.getter_error.emit(e)
        finally:
            self.getter_finished.emit()


    @asyncSlot()
    async def write(self):
        self.setter_started.emit()
        try:
            if self.value.currentText() == '':
                await self._param.set(None)
            await self._param.set(self._param.values[self.value.currentIndex()])
        except Exception as e:
            self.setter_error.emit(e)
        finally:
            self.setter_finished.emit()
