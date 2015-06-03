from PyQt4 import QtCore, QtGui, Qwt5
from concert.quantities import q
from concert.devices.base import Device
from concert.base import HardLimitError
from concert.async import dispatcher
from concert.processes import common, beamline
from concert.helpers import Numeric
from functools import partial
from pyqtgraph.ptime import time
import pyqtgraph as pg
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# from libtiff import TIFF
import scipy.ndimage
import vtk
import numpy as np
import inspect
import weakref


class WidgetPattern(QtGui.QGroupBox):

    """Determines basic device widgets behavior"""
    shadow_accepted = False
    widget_moved = QtCore.pyqtSignal()
    widget_pressed = QtCore.pyqtSignal()
    data_changed = QtCore.pyqtSignal()
    instances = weakref.WeakSet()
    grid_x_step = 16
    grid_y_step = 16

    def __init__(self, name, parent=None, deviceObject=None):
        super(WidgetPattern, self).__init__(parent=parent)
        WidgetPattern.instances.add(self)
        self._green = "background-color: rgb(230, 255, 230);"
        self._orange = "background-color: rgb(255, 214, 156);"
        self._red = "background-color: rgb(255, 153, 153);"
        self._line_start_position = QtCore.QPoint()
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 24, 0, 0)
        self._offset = 0
        self._cursor = QtGui.QCursor
        self.widgetLength = 280
        self.widgetHeight = 20
        self.name = QtGui.QLabel(parent=self)
        self.name.setText(name)
        self.name.adjustSize()
        self.close_button = QtGui.QToolButton(parent=self)
        self.close_button.resize(16, 16)
        self.close_button.setIcon(QtGui.QIcon.fromTheme("window-close"))

        self._units_dict = {}
        self._units_dict['millimeter'] = ["mm", "um"]
        self._units_dict['degree'] = ["deg", "rad"]
        self._units_dict['millimeter / second'] = ["m/s", "mm/s", "um/s"]
        self._units_dict['degree / second'] = ["deg/s", 'rad/s']
        self._units_dict['second'] = ["s", "ms", "us"]
        self._units_dict['pixel'] = ["pixel"]
        self._units_dict['micrometer'] = ["um", "nm"]
        self._units_dict['1 / second'] = ["1 / second"]
        if deviceObject is not None:
            dispatcher.subscribe(deviceObject, "value_changed", self.callback)
        if self.parent() is not None:
            self.close_button.clicked.connect(
                self.parent()._close_button_clicked)

    def mousePressEvent(self, event):
        self._offset = event.pos()
        self.widget_pressed.connect(self.parent().set_current_widget)
        self.widget_pressed.emit()

    def mouseMoveEvent(self, event):
        if self._offset is not None:
            try:
                self.move_widget(QtGui.QGroupBox.mapToParent(self, event.pos() - self._offset))
            except:
                QtGui.QApplication.restoreOverrideCursor()
            else:
                self.widget_moved.emit()

    def mouseReleaseEvent(self, event):
        super(WidgetPattern, self).mouseReleaseEvent(event)
        if WidgetPattern.shadow_accepted:
            self.move_by_grid()
            WidgetPattern.shadow_accepted = False
        QtGui.QApplication.restoreOverrideCursor()
        self._offset = None

    def move_widget(self, position):
        try:
            self.move(position)
        except:
            QtGui.QApplication.restoreOverrideCursor()
            WidgetPattern.shadow_accepted = False
        else:
            WidgetPattern.shadow_accepted = True

    def move_by_grid(self):
        x, y = self.get_grid_position()
        self.move_widget(QtCore.QPoint(x, y))
        QtGui.QApplication.restoreOverrideCursor()
        WidgetPattern.shadow_accepted = False
        self.widget_moved.emit()
        self.parent().save_layout("autosave")

    def get_grid_position(self):
        x = self.mapToParent(QtCore.QPoint(0, 0)).x()
        y = self.mapToParent(QtCore.QPoint(0, 0)).y()
        if x < 0:
            x = 0
        x = (x / WidgetPattern.grid_x_step * WidgetPattern.grid_x_step)
        y = y / WidgetPattern.grid_y_step * WidgetPattern.grid_y_step
        return x, y

    def get_shadow_status(self):
        return WidgetPattern.shadow_accepted

    def get_draw_line_status(self):
        return PortWidget.draw_new_line

    def get_start_line_point(self):
        return self.mapToParent(self._line_start_position)

    def resizeEvent(self, event):
        self.name.move(self.width() / 2 - self.name.width() / 2, 5)
        self.close_button.move(self.width() - self.close_button.width(), 0)

    @classmethod
    def get_instances(self):
        # Returns list of all current instances
        return list(WidgetPattern.instances)

    def callback(self, sender):
        self.data_changed.emit()


class PortWidget(QtGui.QCheckBox):
    draw_new_line = False
    __name__ = "PortWidget"
    port_connected = QtCore.pyqtSignal()
    port_disconnected = QtCore.pyqtSignal()

    def __init__(self, parent, parameter=""):
        super(PortWidget, self).__init__(parameter, parent)
        self.setCheckable(False)
        self.setMouseTracking(True)
        self.is_start_point = False
        self.connection_point = QtCore.QPoint(8, 8)
        self.parameter = parameter
        self.gui = self.parent().parent()
        self.parent().close_button.clicked.connect(self.remove_connection)
        self.parent().widget_moved.connect(self.move_connections)

    def get_line_number(self):
        line_numbers = []
        for number, line in self.gui.lines_info.iteritems():
            if line.start_port == self or line.finish_port == self:
                line_numbers.append(number)
        return line_numbers

    def move_connections(self):
        line_numbers = self.get_line_number()
        for index in line_numbers:
            line = self.gui.lines_info[index]
            if self.is_start_point:
                line.setP1(self.mapTo(self.gui, self.connection_point))
            else:
                line.setP2(self.mapTo(self.gui, self.connection_point))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.pressed.connect(
                partial(
                    self.parent().parent().new_connection,
                    self.mapTo(
                        self.gui,
                        self.connection_point)))
            super(PortWidget, self).mousePressEvent(event)
            PortWidget.draw_new_line = True
        elif event.button() == QtCore.Qt.RightButton:
            self.remove_connection()

    def remove_connection(self):
        line_numbers = self.get_line_number()
        for index in line_numbers:
            self.port_disconnected.emit()
            self.get_another_widget().port_disconnected.emit()
            del self.gui.lines_info[index]

    def mouseReleaseEvent(self, event):
        PortWidget.draw_new_line = False
        super(PortWidget, self).mouseReleaseEvent(event)

    def setLayoutDirection(self, direction):
        super(PortWidget, self).setLayoutDirection(direction)
        self.adjustSize()
        self.connection_point = QtCore.QPoint(self.width() - 8, 8)

    def get_another_widget(self):
        line_numbers = self.get_line_number()
        for index in line_numbers:
            if self.is_start_point:
                return self.gui.lines_info[index].finish_port
            else:
                return self.gui.lines_info[index].start_port
        else:
            return None


class LightSourceWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(LightSourceWidget, self).__init__(name, parent)
        self._port = PortWidget(self, "intensity")
        self.object = deviceObject
        self._spin_value = QtGui.QDoubleSpinBox()
        self._spin_value.setRange(-1000000, 1000000)
        self._spin_value.setDecimals(3)
        self._spin_value.setAccelerated(True)
        self._spin_value.setAlignment(QtCore.Qt.AlignRight)
        self._intensity_units = QtGui.QComboBox()
        self._intensity_units.addItems(["kV", "V", "mV"])
        self._intensity_units.setCurrentIndex(1)
        self._layout = QtGui.QGridLayout()
        self._layout.addWidget(self._port, 0, 0)
        self._layout.addWidget(self._spin_value, 0, 2)
        self._layout.addWidget(self._intensity_units, 0, 3)
        self.resize(self.widgetLength, 60)
        self.layout.addLayout(self._layout)
        self._intensity_units.currentIndexChanged.connect(self._unit_changed)
        self._spin_value.valueChanged.connect(self._number_changed)
        self._unit_changed(1)

    def _unit_changed(self, index):
        self._spin_value.valueChanged.disconnect(self._number_changed)
        self._unit = self._intensity_units.currentText()
        if not self._unit == q.get_symbol(str(self.object.intensity.units)):
            _new_value = self.object.intensity.to(q[str(self._unit)])
        else:
            _new_value = self.object.intensity
        self._spin_value.setValue(float(_new_value.magnitude))
        self._spin_value.valueChanged.connect(self._number_changed)

    def _number_changed(self):
        num = self._spin_value.text()
        unit = self._intensity_units.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.object.intensity = new_value


class MotorWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(MotorWidget, self).__init__(name, parent, deviceObject)

        self.object = deviceObject
        self._home_button = QtGui.QToolButton()
        self._home_button.setIcon(QtGui.QIcon.fromTheme("go-home"))
        self._stop_button = QtGui.QToolButton()
        self._stop_button.setIcon(QtGui.QIcon.fromTheme("process-stop"))
        self._home_button.clicked.connect(self._home_button_clicked)
        self._stop_button.clicked.connect(self._stop_button_clicked)
        self._state = QtGui.QLabel("state")
        self._state_label = QtGui.QLabel("")
        self._state_label.setFrameShape(QtGui.QFrame.WinPanel)
        self._state_label.setFrameShadow(QtGui.QFrame.Raised)
        self._layout = QtGui.QGridLayout()
        self._layout.addWidget(self._state, 0, 0)
        self._layout.addWidget(
            self._state_label,
            0,
            3,
            1,
            1,
            QtCore.Qt.AlignCenter)
        self._layout.addWidget(self._home_button, 1, 1)
        self._layout.addWidget(self._stop_button, 1, 2)
        self._row_number = 1

        """_obj_dict is a dictionary where key is a name of widget,
            value[0] is QtGui.QDoubleSpinBox for this parameter
            value[1] is QtGui.QComboBox object with units for this parameter
            value[2] is PortWidget for this parameter"""

        self._obj_dict = {}
        for param in self.object:
            if not param.name == "state":
                _parameter_name = param.name
                self._value = str(param.get().result()).split(" ", 1)[0]
                self._unit = str(param.get().result()).split(" ", 1)[1]
                parameter_label = PortWidget(self, _parameter_name)
                parameter_label.setFixedWidth(100)
                parameter_value = QtGui.QDoubleSpinBox()
                parameter_value.setRange(-1000000, 1000000)
                parameter_value.setAccelerated(True)
                parameter_value.setDecimals(2)
                parameter_value.setAlignment(QtCore.Qt.AlignRight)
                parameter_unit = QtGui.QComboBox()
                parameter_unit.addItems(
                    self._units_dict[str(getattr(self.object, _parameter_name).units)])
                parameter_unit.setObjectName(_parameter_name)
                self._obj_dict[_parameter_name] = [
                    parameter_value,
                    parameter_unit,
                    parameter_label]
                self._layout.addWidget(parameter_label, self._row_number, 0)
                self._layout.addWidget(parameter_value, self._row_number, 3)
                self._layout.addWidget(parameter_unit, self._row_number, 4)
                parameter_value.valueChanged.connect(self._value_changed)
                parameter_unit.currentIndexChanged.connect(
                    self._get_value_from_concert)
                self._row_number += 1
        self.resize(self.widgetLength, 60 + (self._row_number - 1) * 25)
        self.layout.addLayout(self._layout)
        self._get_value_from_concert()
        self.data_changed.connect(self._get_value_from_concert)

    def _value_changed(self):
        sender = self.sender()
        for key, value in self._obj_dict.iteritems():
            if value[0] == sender:
                num = sender.text()
                unit = value[1].currentText()
                value[2].data = float(num)
                new_value = q.parse_expression(str(num) + str(unit))
                f = getattr(self.object, "set_" + key, None)(new_value)
                f.add_done_callback(self.state_switched)

    def state_switched(self, f):
        QtCore.QObject.connect(
            self,
            QtCore.SIGNAL("state_changed"),
            self._check_state)
        self.emit(QtCore.SIGNAL("state_changed"))

    def _get_value_from_concert(self):
        for _key in self._obj_dict.keys():
            parameter_value = self._obj_dict[_key][0]
            parameter_unit = self._obj_dict[_key][1]
            parameter_value.valueChanged.disconnect(self._value_changed)
            self._unit = parameter_unit.currentText()
            if not q.parse_expression(str(self._unit)) == q.parse_expression(
                    str(getattr(self.object, _key).units)):
                new_value = getattr(self.object, _key).to(
                    q.parse_expression(str(self._unit)))
            else:
                new_value = getattr(self.object, _key)
            parameter_value.setValue(float(new_value.magnitude))
            parameter_value.valueChanged.connect(self._value_changed)
            self._obj_dict[_key][2].data = float(new_value.magnitude)
        self._check_state()

    def _check_state(self):
        state = self.object.state
        if (state == 'standby'):
            self._state_label.setStyleSheet(self._green)
            self._state_label.setText("standby")
        elif (state == 'moving'):
            self._state_label.setStyleSheet(self._orange)
            self._state_label.setText("moving")
        elif (state == 'hard-limit'):
            self._state_label.setStyleSheet(self._red)
            self._state_label.setText("hard-limit")

    def _home_button_clicked(self):
        self.object.home
        self._get_value_from_concert()

    def _stop_button_clicked(self):
        self.object.stop
        self._check_state()


class PositionerWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(PositionerWidget, self).__init__(name, parent)
        self.object = deviceObject
        self._button_left = QtGui.QToolButton()
        self._button_left.setIcon(QtGui.QIcon.fromTheme("go-previous"))
        self._button_left.setAutoRaise(True)
        self._button_right = QtGui.QToolButton()
        self._button_right.setIcon(QtGui.QIcon.fromTheme("go-next"))
        self._button_right.setAutoRaise(True)
        self._button_up = QtGui.QToolButton()
        self._button_up.setIcon(QtGui.QIcon.fromTheme("go-up"))
        self._button_up.setAutoRaise(True)
        self._button_down = QtGui.QToolButton()
        self._button_down.setIcon(QtGui.QIcon.fromTheme("go-down"))
        self._button_down.setAutoRaise(True)
        self._button_back = QtGui.QToolButton()
        self._button_back.setIcon(QtGui.QIcon.fromTheme("go-down"))
        self._button_back.setAutoRaise(True)
        self._button_forward = QtGui.QToolButton()
        self._button_forward.setIcon(QtGui.QIcon.fromTheme("go-up"))
        self._button_forward.setAutoRaise(True)
        self._button_clockwise = QtGui.QToolButton()
        self._button_clockwise.setIcon(
            QtGui.QIcon.fromTheme("object-rotate-right"))
        self._button_clockwise.setAutoRaise(True)
        self._button_counterclockwise = QtGui.QToolButton()
        self._button_counterclockwise.setIcon(
            QtGui.QIcon.fromTheme("object-rotate-left"))
        self._button_counterclockwise.setAutoRaise(True)
        xy_label = QtGui.QLabel("x  y")
        z_label = QtGui.QLabel("z")
        self._rotation_axis = QtGui.QComboBox()
        self._rotation_axis.addItems(["x", "y", "z"])
        self._step_value = QtGui.QDoubleSpinBox()
        self._step_value.setRange(0, 1000000)
        self._step_value.setAccelerated(True)
        self._step_value.setDecimals(2)
        self._step_value.setAlignment(QtCore.Qt.AlignRight)
        self._step_unit = QtGui.QComboBox()
        self._step_unit.addItems(self._units_dict["millimeter"])
        self._rotation_step_value = QtGui.QDoubleSpinBox()
        self._rotation_step_value.setRange(0, 1000000)
        self._rotation_step_value.setAccelerated(True)
        self._rotation_step_value.setDecimals(2)
        self._rotation_step_value.setAlignment(QtCore.Qt.AlignRight)
        self._rotation_step_unit = QtGui.QComboBox()
        self._rotation_step_unit.addItems(self._units_dict["degree"])
        line = QtGui.QFrame()
        line.setGeometry(QtCore.QRect(320, 150, 118, 3))
        line.setFrameShape(QtGui.QFrame.VLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        line1 = QtGui.QFrame()
        line1.setGeometry(QtCore.QRect(320, 150, 118, 3))
        line1.setFrameShape(QtGui.QFrame.VLine)
        line1.setFrameShadow(QtGui.QFrame.Sunken)
        self._layout = QtGui.QGridLayout()
        self._layout.addWidget(self._button_left, 1, 0)
        self._layout.addWidget(self._button_right, 1, 2)
        self._layout.addWidget(self._button_forward, 0, 1)
        self._layout.addWidget(self._button_back, 2, 1)
        self._layout.addWidget(self._button_up, 0, 4, QtCore.Qt.AlignBottom)
        self._layout.addWidget(self._button_down, 2, 4, QtCore.Qt.AlignTop)
        self._layout.addWidget(
            self._button_clockwise,
            1,
            10,
            QtCore.Qt.AlignLeft)
        self._layout.addWidget(
            self._button_counterclockwise,
            1,
            7,
            QtCore.Qt.AlignRight)
        self._layout.addWidget(self._step_value, 4, 0, 1, 3)
        self._layout.addWidget(self._step_unit, 4, 3, 1, 2)
        self._layout.addWidget(self._rotation_step_value, 4, 7, 1, 2)
        self._layout.addWidget(self._rotation_step_unit, 4, 9, 1, 2)
        self._layout.addWidget(xy_label, 1, 1)
        self._layout.addWidget(z_label, 1, 4, QtCore.Qt.AlignCenter)
        self._layout.addWidget(
            self._rotation_axis,
            1,
            8,
            1,
            2,
            QtCore.Qt.AlignCenter)
        self._layout.addWidget(line, 0, 5, 10, 1)
        self.layout.addLayout(self._layout)
        self.setFixedSize(self.widgetLength, 140)
        self._button_left.pressed.connect(self._button_left_clicked)
        self._button_right.pressed.connect(self._button_right_clicked)
        self._button_up.clicked.connect(self._button_up_clicked)
        self._button_down.clicked.connect(self._button_down_clicked)
        self._button_forward.clicked.connect(self._button_forward_clicked)
        self._button_back.clicked.connect(self._button_back_clicked)
        self._button_clockwise.clicked.connect(self._button_clockwise_clicked)
        self._button_counterclockwise.clicked.connect(
            self._button_counterclockwise_clicked)

    def _button_left_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.object.left(new_value)

    def _button_right_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.object.right(new_value)

    def _button_up_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.object.up(new_value)

    def _button_down_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.object.down(new_value)

    def _button_forward_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.object.forward(new_value)

    def _button_back_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.object.back(new_value)

    def _button_clockwise_clicked(self):
        pass

    def _button_counterclockwise_clicked(self):
        pass


class ShutterWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(ShutterWidget, self).__init__(name, parent)
        self.object = deviceObject
        self._port = PortWidget(self, "state")
        self._slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMaximumWidth(50)
        self._slider.setMaximum(1)
        on_label = QtGui.QLabel("On")
        on_label.adjustSize()
        off_label = QtGui.QLabel("Off")
        off_label.adjustSize()
        self._layout = QtGui.QGridLayout()
        self._layout.addWidget(self._port, 0, 0, QtCore.Qt.AlignLeft)
        self._layout.addWidget(on_label, 0, 3, QtCore.Qt.AlignRight)
        self._layout.addWidget(off_label, 0, 1, QtCore.Qt.AlignLeft)
        self._layout.addWidget(self._slider, 0, 2)
        self._slider.valueChanged.connect(self._slider_value_changed)
        self.setFixedSize(self.widgetLength, 60)
        self.layout.addLayout(self._layout)
        self._get_state_from_concert()
        dispatcher.subscribe(deviceObject, "state_changed", self._callback)

    def _get_state_from_concert(self):
        if self.object.state == 'open':
            self._slider.setValue(1)
        else:
            self._slider.setValue(0)

    def _slider_value_changed(self):
        value = self._slider.value()
        if value == 1:
            self.object.open()
        elif value == 0:
            self.object.close()

    def _callback(self, sender):
        self._get_state_from_concert()


class CameraWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(CameraWidget, self).__init__(name, parent)
        self.object = deviceObject
        layout = QtGui.QGridLayout()
        self.imv = pg.ImageView(self)
        frame = self.object.grab()
        self.imv.setImage(frame, autoRange=False)
        self.layout.addWidget(self.imv)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self._row_number = 3
        self._port_out = PortWidget(self, "port_out")
        self._start_recording_button = QtGui.QToolButton()
        self._start_recording_button.setIcon(QtGui.QIcon.fromTheme("media-playback-start"))
        self._start_recording_button.clicked.connect(self.start_recording)
        self._stop_recording_button = QtGui.QToolButton()
        self._stop_recording_button.setIcon(QtGui.QIcon.fromTheme("media-playback-stop"))
        self._stop_recording_button.clicked.connect(self.stop_recording)
        hlayout = QtGui.QHBoxLayout()
        hlayout.addWidget(self._port_out, 1, QtCore.Qt.AlignLeft)
        hlayout.addWidget(self._start_recording_button, 2)
        hlayout.addWidget(self._stop_recording_button, 3)
        self.layout.addLayout(hlayout)
        self.widget = QtGui.QWidget()
        self._obj_dict = {}
        for param in self.object:
            _parameter_name = param.name
            if ' ' in str(param.get().result()):
                parameter_label = QtGui.QLabel(_parameter_name)
                parameter_value = QtGui.QDoubleSpinBox()
                parameter_value.setRange(-1000000, 1000000)
                parameter_value.setAccelerated(True)
                parameter_value.setDecimals(3)
                parameter_value.setAlignment(QtCore.Qt.AlignRight)
                parameter_unit = QtGui.QComboBox()
                parameter_unit.addItems(
                    self._units_dict[str(getattr(self.object, _parameter_name).units)])
                parameter_unit.setObjectName(_parameter_name)
                self._obj_dict[_parameter_name] = [
                    parameter_value,
                    parameter_unit]
                layout.addWidget(parameter_label, self._row_number, 0)
                layout.addWidget(parameter_value, self._row_number, 3)
                layout.addWidget(parameter_unit, self._row_number, 4)
                parameter_value.valueChanged.connect(self._value_changed)
                parameter_unit.currentIndexChanged.connect(
                    self._get_value_from_concert)
                self._row_number += 1
        self.setFixedSize(500, 500)
        self.widget.setLayout(layout)
        scroll = QtGui.QScrollArea()
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(150)
        scroll.setWidget(self.widget)
        self.layout.addWidget(scroll)
        self._get_value_from_concert()

    def update(self, fps=None):
        frame = self.object.grab()
        self._port_out.data = frame
        now = time()
        self.imv.setImage(frame, autoRange=False, autoLevels=False)

    def _value_changed(self):
        sender = self.sender()
        for key, value in self._obj_dict.iteritems():
            if value[0] == sender:
                num = sender.text()
                unit = value[1].currentText()
                new_value = q.parse_expression(str(num) + str(unit))
                try:
                    getattr(
                        self.object, "set_" + key, None)(new_value)
                except HardLimitError:
                    pass
                self._get_value_from_concert()

    def _get_value_from_concert(self):
        for _key in self._obj_dict.keys():
            parameter_value = self._obj_dict[_key][0]
            parameter_unit = self._obj_dict[_key][1]
            parameter_value.valueChanged.disconnect(self._value_changed)
            self._unit = parameter_unit.currentText()
            if not q.parse_expression(str(self._unit)) == q.parse_expression(
                    str(getattr(self.object, _key).units)):
                new_value = getattr(self.object, _key).to(
                    q.parse_expression(str(self._unit)))
            else:
                new_value = getattr(self.object, _key)
            parameter_value.setValue(float(new_value.magnitude))
            parameter_value.valueChanged.connect(self._value_changed)

    def start_recording(self):
        try:
            self.object.start_recording()
        except Exception, msg:
            print msg
        self.timer.start(0)

    def stop_recording(self):
        try:
            self.object.stop_recording()
        except Exception, msg:
            print msg
        self.timer.stop()

    def close(self):
        self.timer.stop()
        super(CameraWidget, self).close()
        self.destroy()


class FunctionWidget(WidgetPattern):
    callback_signal = QtCore.pyqtSignal()

    def __init__(self, name, parent=None):
        super(FunctionWidget, self).__init__(name, parent)
        self._layout = QtGui.QGridLayout()
        self._state_label = QtGui.QLabel("state")
        self._state_label.setFrameShape(QtGui.QFrame.WinPanel)
        self._state_label.setFrameShadow(QtGui.QFrame.Raised)
        self._layout.addWidget(self._state_label, 0, 1, 1, 2, QtCore.Qt.AlignCenter)
        if hasattr(common, name):
            self.func = getattr(common, name)
        elif hasattr(beamline, name):
            self.func = getattr(beamline, name)
        else:
            return
        for i in xrange(len(self.func.e_args)):
            if inspect.isclass(self.func.e_args[i]) and issubclass(
                    self.func.e_args[i], Device):
                exec(
                    "self.%s = PortWidget(self, '%s')" %
                    (str(
                        self.func.f_args[i]),
                        self.func.f_args[i]))
                port = getattr(self, self.func.f_args[i])
                port.setLayoutDirection(QtCore.Qt.RightToLeft)
                self._layout.addWidget(port, i+1, 1, 1, 2, QtCore.Qt.AlignLeft)

            elif isinstance(self.func.e_args[i], Numeric):
                exec(
                    "self.%s_label = QtGui.QLabel('%s (%s)')" %
                    (str(
                        self.func.f_args[i]), self.func.f_args[i], str(
                        self.func.e_args[i].units or "no unit")))
                label = getattr(self, str(self.func.f_args[i]) + "_label")
                self._layout.addWidget(label, i+1, 0)
                for j in xrange(self.func.e_args[i].dimension):
                    exec(
                        "self.%s%i = QtGui.QDoubleSpinBox()" %
                        (str(
                            self.func.f_args[i]),
                            j))
                    spin_box = getattr(self, str(self.func.f_args[i]) + str(j))
                    spin_box.setRange(-1000, 1000)
                    spin_box.setValue(0)
                    self._layout.addWidget(spin_box, i+1, j + 1)
            else:
                print "Sorry, I didn't finish this part yet"

        for j in self.func.e_keywords:
            if inspect.isclass(self.func.e_keywords[j]) and issubclass(
                    self.func.e_keywords[j], Device):
                i += 1
                exec("self.%s = PortWidget(self, '%s')" % (str(j), j))
                port = getattr(self, j)
                port.setLayoutDirection(QtCore.Qt.RightToLeft)
                self._layout.addWidget(port, i+1, 1, 1, 2, QtCore.Qt.AlignLeft)

        for j in self.func.e_keywords:
            if isinstance(self.func.e_keywords[j], Numeric) and not j == "output":
                i += 1
                exec(
                    "self.%s_label = QtGui.QLabel('%s (%s)')" %
                    (str(j), j, str(
                        self.func.e_keywords[j].units or "no unit")))
                label = getattr(self, str(j) + "_label")
                self._layout.addWidget(label, i+1, 0)
                for k in xrange(self.func.e_keywords[j].dimension):
                    exec(
                        "self.%s%i = QtGui.QDoubleSpinBox()" %
                        (str(j), k))
                    spin_box = getattr(self, str(j) + str(k))
                    spin_box.setRange(-1000, 1000)
                    self._layout.addWidget(spin_box, i+1, 1 + k)

        self._play_button = QtGui.QToolButton()
        self._play_button.setIcon(QtGui.QIcon.fromTheme("media-playback-start"))
        self._play_button.setText("play")
        self._play_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._play_button.clicked.connect(self.play_button_clicked)
        self._play_button.adjustSize()
        self._layout.addWidget(self._play_button, 99, 0, 1, 3, QtCore.Qt.AlignCenter)
        self.layout.addLayout(self._layout)
        self.adjustSize()
        self.gui = self.parent()
        self.callback_signal.connect(self._set_done_state)

    def play_button_clicked(self):
        args = []
        for i in xrange(len(self.func.e_args)):
            if inspect.isclass(self.func.e_args[i]) and issubclass(
                    self.func.e_args[i], Device):
                port = getattr(self, self.func.f_args[i])
                if port.get_another_widget() is not None:
                    args.append(
                        getattr(port.get_another_widget().parent(), "object"))
                else:
                    print "%s is not defined!" % port.text()
                    return 0
            elif isinstance(self.func.e_args[i], Numeric):
                spin_box = getattr(self, self.func.f_args[i] + "0")

                if self.func.e_args[i].units is None:
                    args.append(spin_box.value())
                    continue

                if self.func.e_args[i].dimension == 1:
                    args.append(
                        q[str(spin_box.value()) + str(self.func.e_args[i].units.units)])
                else:
                    list = []
                    units = self.func.e_args[i].units
                    list.append(spin_box.value())
                    for j in xrange(1, self.func.e_args[i].dimension):
                        value = getattr(
                            self, str(
                                self.func.f_args[i]) + str(j)).value()
                        list.append(value)
                    args.append(list * units)

        for i in xrange(len(self.func.e_args), len(self.func.f_args)):
            item = self.func.f_args[i]
            if inspect.isclass(self.func.e_keywords[item]) and issubclass(
                    self.func.e_keywords[item], Device):
                port = getattr(self, item)
                if port.get_another_widget() is not None:
                    args.append(
                        getattr(port.get_another_widget().parent(), "object"))
                else:
                    break

            elif isinstance(self.func.e_keywords[item], Numeric):
                spin_box = getattr(self, item + "0")
                if self.func.e_keywords[item].units is None:
                    args.append(spin_box.value())
                    continue

                if self.func.e_keywords[item].dimension == 1:
                    args.append(
                        q[str(spin_box.value()) + str(self.func.e_keywords[item].units.units)])
                else:
                    list = []
                    units = self.func.e_keywords[item].units
                    list.append(spin_box.value())
                    for j in xrange(1, self.func.e_keywords[item].dimension):
                        value = getattr(self, str(item) + str(j)).value()
                        list.append(value)
                    args.append(list * units)
            else:
                args.append(self.func.f_defaults[i - len(self.func.e_args)])
        try:
            f = self.func(*args)
            f.add_done_callback(self._callback)
        except Exception as e:
            print "Error: ", str(e)
            self._state_label.setStyleSheet(self._red)
            self._state_label.setText("error")
        else:
            self._state_label.setStyleSheet(self._orange)
            self._state_label.setText("processing")

    def _callback(self, sender):
        self.callback_signal.emit()

    def _set_done_state(self):
        self._state_label.setStyleSheet(self._green)
        self._state_label.setText("done")


class VisualizationWidget(QtGui.QGroupBox):

    def __init__(self, name, parent=None):
        super(VisualizationWidget, self).__init__(parent=parent)
        self._line_start_position = QtCore.QPoint()
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 24, 0, 0)
        self._offset = 0
        self._cursor = QtGui.QCursor
        self.widgetLength = 280
        self.widgetHeight = 20
        self.name = QtGui.QLabel(parent=self)
        self.name.setText(name)
        self.name.adjustSize()
        self.close_button = QtGui.QToolButton(parent=self)
        self.close_button.resize(16, 16)
        self.close_button.setIcon(QtGui.QIcon.fromTheme("window-close"))
        self.resize(200, 200)
        self.path = "/home"
        self.setupUi()
        self.rec = MeshGenerator()
        self.isolavel_slider.setValue(self.rec.isolevel)
        self.gaussian_slider.setValue(self.rec.gaussian_param)
        self.erosion_iterations_slider.setValue(self.rec.erosion_iterations)
        self.detalization_slider.setValue(self.rec.detalization)

        self.thread = QtCore.QThread()
        self.rec.moveToThread(self.thread)
        self.thread.started.connect(self.rec.reconstruction_from_array)
        self.rec.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.update_render)

        self.close_button.clicked.connect(self.close)

    def resizeEvent(self, event):
        self.name.move(self.width() / 2 - self.name.width() / 2, 5)
        self.close_button.move(
            self.width() -
            self.close_button.width(),
            0)

    def setupUi(self):
        self.vtkWidget = VtkRender(self)
        self.layout.addWidget(self.vtkWidget, 0)
        panel = self.create_panel()
        self.layout.addLayout(panel, 1)
        self.resize(500, 500)

    def create_panel(self):
        layout = QtGui.QGridLayout()
        self.isolavel_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.isolavel_slider.setRange(-300, 300)
        self.isolavel_label = QtGui.QLabel("isolevel")
        self.isolavel_value = QtGui.QLabel(str(self.isolavel_slider.value()))
        self.isolavel_value.setFixedWidth(29)
        self.isolavel_slider.valueChanged.connect(self.isolavel_changed)
        self.gaussian_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.gaussian_slider.setRange(0, 50)
        self.gaussian_filter_label = QtGui.QLabel("blur")
        self.gaussian_filter_value = QtGui.QLabel(
            str(self.gaussian_slider.value()))
        self.gaussian_slider.valueChanged.connect(self.gaussian_filter_changed)
        self.erosion_iterations_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.erosion_iterations_slider.setRange(0, 50)
        self.erosion_iterations_slider.valueChanged.connect(
            self.erosion_iterations_changed)
        self.erosion_iterations_label = QtGui.QLabel("roughness")
        self.erosion_iterations_value = QtGui.QLabel(
            str(self.erosion_iterations_slider.value()))

        self.detalization_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.detalization_slider.setRange(4, 10)
        self.detalization_label = QtGui.QLabel("level of detail")
        self.detalization_value = QtGui.QLabel(
            str(self.isolavel_slider.value()))
        self.detalization_slider.valueChanged.connect(
            self.detalization_changed)

        self.apply_button = QtGui.QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_clicked)

        self.load_files_button = QtGui.QPushButton("from files")
        self.load_files_button.clicked.connect(self.load_files_clicked)

        self.pb = QtGui.QProgressBar()
        self.pb.setRange(0, 0)
        self.pb.hide()
        layout.addWidget(self.isolavel_slider, 0, 1)
        layout.addWidget(self.isolavel_label, 0, 0)
        layout.addWidget(self.isolavel_value, 0, 2)
        layout.addWidget(self.gaussian_slider, 1, 1)
        layout.addWidget(self.gaussian_filter_label, 1, 0)
        layout.addWidget(self.gaussian_filter_value, 1, 2)
        layout.addWidget(self.erosion_iterations_slider, 2, 1)
        layout.addWidget(self.erosion_iterations_label, 2, 0)
        layout.addWidget(self.erosion_iterations_value, 2, 2)

        layout.addWidget(self.detalization_slider, 3, 1)
        layout.addWidget(self.detalization_label, 3, 0)
        layout.addWidget(self.detalization_value, 3, 2)

        layout.addWidget(self.apply_button, 4, 0, 1, 3, QtCore.Qt.AlignCenter)
        layout.addWidget(self.load_files_button, 4, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.pb, 4, 1, 1, 2, QtCore.Qt.AlignRight)
        return layout

    def isolavel_changed(self, value):
        self.isolavel_value.setText(str(value))

    def gaussian_filter_changed(self, value):
        self.gaussian_filter_value.setText(str(value))

    def erosion_iterations_changed(self, value):
        self.erosion_iterations_value.setText(str(value))

    def detalization_changed(self, value):
        self.detalization_value.setText(
            "%.0f" % ((10. / (11 - value)) ** 2) + " %")

    def set_reconstruction_parameters(self):
        self.rec.isolevel = self .isolavel_slider.value()
        self.rec.gaussian_param = self.gaussian_slider.value()
        self.rec.erosion_iterations = self.erosion_iterations_slider.value()
        self.rec.detalization = self.detalization_slider.value()

    def apply_clicked(self):
        self.pb.show()
        self.set_reconstruction_parameters()
        self.thread.start()

    def load_files_clicked(self):
        self.set_reconstruction_parameters()
        files = QtGui.QFileDialog.getOpenFileNames(
            self,
            'Open files',
            '%s' % self.path,
            "All files (*.*);;JPEG (*.jpg *.jpeg);;TIFF (*.tif)",
            "TIFF (*.tif)")
        if len(files) > 1:
            self.path = QtCore.QFileInfo(files[0]).path()
            file_names = []
            for i, name in enumerate(files):
                file_names.append(str(files.__getitem__(i)))
            self.rec.files = file_names
            self.thread.started.disconnect()
            self.thread.started.connect(self.rec.reconstruction_from_files)
            self.thread.start()
            self.pb.show()

    def update_render(self):
        self.vtkWidget.show(self.rec.mesh)
        self.pb.hide()


class VtkRender(QVTKRenderWindowInteractor):

    def __init__(self, parent):
        super(VtkRender, self).__init__(parent)

    def show(self, mesh):
        geoBoneMapper = vtk.vtkPolyDataMapper()
        geoBoneMapper.SetInputConnection(mesh.GetOutputPort())
        geoBoneMapper.ScalarVisibilityOff()
        actorBone = vtk.vtkActor()
        actorBone.SetMapper(geoBoneMapper)
        actor = actorBone
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.329412, 0.34902, 0.427451)
        self.GetRenderWindow().AddRenderer(ren)
        iren = self.GetRenderWindow().GetInteractor()
        ren.AddActor(actor)
        iren.Initialize()


class MeshGenerator(QtCore.QObject):

    mesh = None
    finished = QtCore.pyqtSignal()
    isolevel = 115
    gaussian_param = 6
    erosion_iterations = 10
    detalization = 5
    files = None
    array = None

    def reconstruction_from_files(self):
        array = self.array_from_files(self.files)
        data = self.numpy_to_vtk(array)
        self.mesh = self.make_iso(data)
        self.finished.emit()

    def reconstruction_from_array(self):
        n = 256
        if self.array is None:
            self.array = self.make_cube(n, n / 2)
        data = self.numpy_to_vtk(self.array)
        self.mesh = self.make_iso(data)
        self.finished.emit()

    def make_sphere(self, n, radius):
        z, y, x = np.mgrid[-n / 2:n / 2, -n / 2:n / 2, -n / 2:n / 2]
        distances = x ** 2 + y ** 2 + z ** 2
        sphere = np.zeros_like(distances)
        sphere[distances <= radius ** 2] = 1
        return sphere.astype(np.ubyte)

    def make_cube(self, n, edge):
        cube = np.zeros((n, n, n), dtype=np.ubyte)
        cube[n / 2 - edge / 2:n / 2 + edge / 2,
             n / 2 - edge / 2:n / 2 + edge / 2,
             n / 2 - edge / 2:n / 2 + edge / 2] = 1
        return cube

    def array_from_files(self, files):
        frame_rate = 11 - self.detalization
        (x, y) = TIFF.open(files[0]).read_image()[::frame_rate, ::frame_rate].shape
        sample = np.zeros((x, y, len(files[::frame_rate])))
        zeros = np.zeros((x, y))
        for i in xrange(0, len(files), frame_rate):
            im = TIFF.open(files[i])
            image = im.read_image()
            gauss_denoised = scipy.ndimage.gaussian_filter(
                image,
                self.gaussian_param)
            filtered = gauss_denoised < self.isolevel
            eroded_img = scipy.ndimage.binary_erosion(
                filtered,
                iterations=self.erosion_iterations)
            reconstruct_img = scipy.ndimage.binary_propagation(
                eroded_img,
                mask=filtered)
            tmp = np.logical_not(reconstruct_img)
            eroded_tmp = scipy.ndimage.binary_erosion(
                tmp,
                iterations=self.erosion_iterations /
                2)
            reconstruct_final = np.logical_not(
                scipy.ndimage.binary_propagation(
                    eroded_tmp,
                    mask=tmp))
            sample[:, :, int(i / frame_rate)] = reconstruct_final[::frame_rate, ::frame_rate]
        return sample.astype(np.ubyte)

    def numpy_to_vtk(self, array):
        dataImporter = vtk.vtkImageImport()
        data_string = array.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        z, y, x = [dim - 1 for dim in array.shape]
        dataImporter.SetDataExtent(0, x, 0, y, 0, z)
        dataImporter.SetWholeExtent(0, x, 0, y, 0, z)
        return dataImporter

    def make_iso(self, data):
        contourBoneHead = vtk.vtkDiscreteMarchingCubes()
        contourBoneHead.SetInput(data.GetOutput())
        contourBoneHead.ComputeNormalsOn()
        contourBoneHead.ComputeScalarsOn()
        contourBoneHead.SetValue(0, 1)
        contourBoneHead.Update()

        poly = vtk.vtkPolyData()
        poly.DeepCopy(contourBoneHead.GetOutput())

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInput(poly)
        smoother.GenerateErrorScalarsOn()
        smoother.SetNumberOfIterations(50)
        smoother.SetRelaxationFactor(0.05)
        smoother.Update()
        return smoother


class PlotWidget(WidgetPattern):

    def __init__(self, name, parent=None):
        super(PlotWidget, self).__init__(name, parent)
        self._layout = QtGui.QGridLayout()
        self._port = PortWidget(self, "in")
        self._port.setLayoutDirection(QtCore.Qt.RightToLeft)
        self._port.adjustSize()
        self._port.port_connected.connect(self.port_connected)
        self._port.port_disconnected.connect(self.port_disconnected)

        self._layout.addWidget(self._port, 1, 1, 1, 2, QtCore.Qt.AlignLeft)
        self.plot = Qwt5.QwtPlot()
        self.plot.setCanvasBackground(QtCore.Qt.white)

        # Initialize data
        self.x = np.arange(0.0, 128)
        self.y = np.zeros(len(self.x))
        self.plot.setAxisScale(Qwt5.QwtPlot.xBottom, 0, len(self.x))

        self.plot.curveR = Qwt5.QwtPlotCurve("states")
        self.plot.curveR.attach(self.plot)
        self.plot.curveR.setPen(QtGui.QPen(QtCore.Qt.red))

        mY = Qwt5.QwtPlotMarker()
        mY.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        mY.setLineStyle(Qwt5.QwtPlotMarker.HLine)
        mY.setYValue(0.0)
        mY.attach(self.plot)

        self.plot.setAxisTitle(Qwt5.QwtPlot.xBottom, "States")
        self.plot.setAxisTitle(Qwt5.QwtPlot.yLeft, "Values")
        self.data_changed.connect(self.replot)
        self._layout.addWidget(self.plot)
        self.layout.addLayout(self._layout)
        self.points_to_show = 1

    def port_connected(self):
        self.connected_device = self.sender().get_another_widget().parent().object
        self.parameter = self.sender().get_another_widget().parameter
        units = self.connected_device[self.parameter].get().result().units
        self.plot.setAxisTitle(Qwt5.QwtPlot.yLeft, "%s, %s" % (self.parameter, units))
        dispatcher.subscribe(self.connected_device, "value_changed", self.callback)

    def port_disconnected(self):
        dispatcher.unsubscribe(self.connected_device, "value_changed", self.callback)
        self.points_to_show = 1

    def replot(self):
        data = self.connected_device[self.parameter].get().result().magnitude
        self.y = np.concatenate((self.y[:1], self.y[:-1]), 1)
        self.y[0] = data
        self.plot.curveR.setData(self.x[:self.points_to_show], self.y[:self.points_to_show])
        self.plot.replot()
        if self.points_to_show < len(self.x):
            self.points_to_show += 1
