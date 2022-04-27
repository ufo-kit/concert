from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout, QLineEdit, \
    QCheckBox, QDoubleSpinBox
from PyQt5.QtCore import QTimer, pyqtSignal
from concert.base import Parameterizable, ParameterValue, QuantityValue, StateValue, SelectionValue
import concert.base
from concert.coroutines.base import run_in_loop
from concert.quantities import q
from qasync import asyncSlot
from concert.gui import with_signals


class ParameterizableWidget(QWidget):
    def __init__(self, parameterizable):
        if not isinstance(parameterizable, concert.base.Parameterizable):
            raise Exception("Only Parameterizables can be wrapped.")

        self._parameterizable = parameterizable
        parameterizable.widget = self
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._params = []
        self.build_layout()
        self._layout.addWidget(PollingWidget(self))

    def build_layout(self):
        for param in self._parameterizable:
            print(param)
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
        self._params.append(widget)
        self._layout.addWidget(widget)

    def add_quantity_to_layout(self, param):
        widget = QuantityWidget(param)
        self._params.append(widget)
        self._layout.addWidget(widget)

    def add_parameter_to_layout(self, param):
        widget = ParameterWidget(param)
        self._params.append(widget)
        self._layout.addWidget(widget)

    def add_selection_to_layout(self, param):
        widget = SelectionWidget(param)
        self._params.append(widget)
        self._layout.addWidget(widget)

    def update(self):
        for param in self._params:
            param.read()

    def polling(self, state):
        for param in self._params:
            if state:
                param.deactivate()
            else:
                param.activate()


class PollingWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self._layout = QHBoxLayout()
        self.setLayout(self._layout)
        self._layout.addWidget(QLabel("Poll"))
        self._interval = QDoubleSpinBox()
        self._interval.setMinimum(0.01)
        self._interval.setMaximum(100.)
        self._interval.setValue(1.0)
        self._polling = QCheckBox()
        self._polling.stateChanged.connect(self.changed_polling)
        self._layout.addWidget(self._polling)
        self._timer = QTimer()
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(parent.update)
        self._layout.addWidget(QLabel("Interval in s:"))
        self._layout.addWidget(self._interval)

    def changed_polling(self, state):
        # 2 == checked??? 0 == unchecked
        if state == 2:
            self._timer.setInterval(self._interval.value() * 1000.)
            self._timer.start()
            self._interval.setEnabled(False)
            self.parent().polling(True)
        if state == 0:
            self._timer.stop()
            self._interval.setEnabled(True)
            self.parent().polling(False)


class ParameterWidget(QWidget):
    setter_started = pyqtSignal()
    setter_finished = pyqtSignal()
    setter_error = pyqtSignal(Exception)

    getter_started = pyqtSignal()
    getter_finished = pyqtSignal()
    getter_error = pyqtSignal(Exception)

    def __init__(self, param):
        super().__init__()
        self._param = param
        self.name_label = QLabel(param.name)
        self.read_button = QPushButton('read')
        self.write_button = QPushButton('write')

        self.read_button.clicked.connect(self.read)
        self.write_button.clicked.connect(self.write)
        self._data_type = type(run_in_loop(self._param.get()))

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
