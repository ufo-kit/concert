from PyQt4 import QtCore, QtGui
from concert.quantities import q
from concert.devices.base import Device
from concert.base import HardLimitError
from concert.async import dispatcher
from concert import processes
from pyqtgraph.ptime import time
import pyqtgraph as pg
from functools import partial


class WidgetPattern(QtGui.QGroupBox):

    """Determines basic device widgets behavior"""
    shadow_accepted = False
    widget_moved = QtCore.pyqtSignal()
    widget_pressed = QtCore.pyqtSignal()

    def __init__(self, name, parent=None):
        super(WidgetPattern, self).__init__(parent=parent)
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
        self.close_button.resize(24, 24)
        self.close_button.setAutoRaise(True)
        self.close_button.setIcon(QtGui.QIcon.fromTheme("application-exit"))
        self.grid_x_step = 70
        self.grid_y_step = 32
        self._units_dict = {}
        self._units_dict['meter'] = ["mm", "um"]
        self._units_dict['degree'] = ["deg", "rad"]
        self._units_dict['meter / second'] = ["m/s", "mm/s", "um/s"]
        self._units_dict['degree / second'] = ["deg/s", 'rad/s']
        self._units_dict['second'] = ["s", "ms", "us"]
        self._units_dict['pixel'] = ["pixel"]
        self._units_dict['micrometer'] = ["um", "nm"]
        self._units_dict['1 / second'] = ["1 / second"]

    def mousePressEvent(self, event):
        self._offset = event.pos()
        self.widget_pressed.connect(self.parent().set_current_widget)
        self.widget_pressed.emit()

    def mouseMoveEvent(self, event):
        if self._offset is not None:
            try:
                self.move_widget(
                    QtGui.QGroupBox.mapToParent(
                        self,
                        event.pos() -
                        self._offset))
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

    def get_grid_position(self):
        x = self.mapToParent(QtCore.QPoint(0, 0)).x()
        y = self.mapToParent(QtCore.QPoint(0, 0)).y()
        if x < self.widgetLength:
            x = self.widgetLength
        x = (x / self.grid_x_step * self.grid_x_step)
        y = y / self.grid_y_step * self.grid_y_step
        return x, y

    def get_shadow_status(self):
        if WidgetPattern.shadow_accepted:
            return True
        else:
            return False

    def get_draw_line_status(self):
        if PortWidget.draw_new_line:
            return True
        else:
            return False

    def get_start_line_point(self):
        return self.mapToParent(self._line_start_position)

    def resizeEvent(self, event):
        self.name.move(self.width() / 2 - self.name.width() / 2, 5)
        self.close_button.move(
            self.width() -
            self.close_button.width(),
            0)


class PortWidget(QtGui.QCheckBox):
    draw_new_line = False
    __name__ = "PortWidget"

    def __init__(self, parent, parameter=""):
        super(PortWidget, self).__init__(parameter, parent)
        self.setCheckable(False)
        self.setMouseTracking(True)
        self.is_start_point = False
        self.connection_point = QtCore.QPoint(8, 8)
        self.dic_index = None
        self.index = None
        self.parameter = parameter
        self.gui = self.parent().parent()
        self.parent().close_button.clicked.connect(self.remove_connection)
        self.parent().widget_moved.connect(self.move_connections)

    def move_connections(self):
        if self.dic_index is not None:
            line = self.gui.lines_info[self.dic_index]
            if self.is_start_point:
                line.setP1(self.mapTo(self.gui, self.connection_point))
            else:
                line.setP2(self.mapTo(self.gui, self.connection_point))

    def mousePressEvent(self, event):
        self.pressed.connect(
            partial(
                self.parent().parent().new_connection,
                self.mapTo(
                    self.gui,
                    self.connection_point)))
        self.remove_connection()
        super(PortWidget, self).mousePressEvent(event)
        PortWidget.draw_new_line = True

    def remove_connection(self):
        if self.dic_index is not None:
            del self.gui.lines_info[self.dic_index]

    def mouseReleaseEvent(self, event):
        PortWidget.draw_new_line = False
        super(PortWidget, self).mouseReleaseEvent(event)

    def setLayoutDirection(self, direction):
        super(PortWidget, self).setLayoutDirection(direction)
        self.adjustSize()
        self.connection_point = QtCore.QPoint(self.width() - 8, 8)

    def get_another_widget(self):
        if self.dic_index is not None:
            if self.is_start_point:
                return self.gui.lines_info[self.dic_index].finish_port.parent()
            else:
                return self.gui.lines_info[self.dic_index].start_port.parent()
        else:
            return None


class LightSourceWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(LightSourceWidget, self).__init__(name, parent)
        self._port = PortWidget(self, "Intensity")
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
        self.setFixedSize(self.widgetLength, 60)
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
        super(MotorWidget, self).__init__(name, parent)
        self._green = "background-color: rgb(230, 255, 230);"
        self._orange = "background-color: rgb(255, 214, 156);"
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
            value[1] is QtGui.QComboBox object with units for this parameter"""

        self._obj_dict = {}
        for param in self.object:
            if not param.name == "state":
                _parameter_name = param.name
                self._value = str(param.get().result()).split(" ", 1)[0]
                self._unit = str(param.get().result()).split(" ", 1)[1]
                parameter_label = PortWidget(self, _parameter_name)
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
                self._layout.addWidget(parameter_label, self._row_number, 0)
                self._layout.addWidget(parameter_value, self._row_number, 3)
                self._layout.addWidget(parameter_unit, self._row_number, 4)
                parameter_value.valueChanged.connect(self._value_changed)
                parameter_unit.currentIndexChanged.connect(
                    self._get_value_from_concert)
                self._row_number += 1
        self.setFixedSize(self.widgetLength, 60 + (self._row_number - 1) * 25)
        self.layout.addLayout(self._layout)
        self._get_value_from_concert()
        dispatcher.subscribe(self.object, 'standby', self.callback)

    def callback(self, who):
        print "called by:"

    def _value_changed(self):
        sender = self.sender()
        for key, value in self._obj_dict.iteritems():
            if value[0] == sender:
                num = sender.text()
                unit = value[1].currentText()
                new_value = q.parse_expression(str(num) + str(unit))
                try:
                    f = getattr(
                        self.object, "set_" + key, None)(new_value)
                    f.add_done_callback(self.state_switched)
                except HardLimitError:
                    pass
                self._get_value_from_concert()
        dispatcher.send(self.object, 'standby')

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
        self._check_state()

    def _check_state(self):
        state = self.object.state
        if (state == 'standby'):
            self._state_label.setStyleSheet(self._green)
            self._state_label.setText("standby")
        elif (state == 'moving'):
            self._state_label.setStyleSheet(self._orange)
            self._state_label.setText("moving")

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
        self._rotation_axis.setMaximumWidth(50)
        self._rotation_axis.setMinimumWidth(35)
        self._step_value = QtGui.QDoubleSpinBox()
        self._step_value.setRange(0, 1000000)
        self._step_value.setAccelerated(True)
        self._step_value.setDecimals(2)
        self._step_value.setAlignment(QtCore.Qt.AlignRight)
        self._step_unit = QtGui.QComboBox()
        self._step_unit.addItems(self._units_dict["meter"])
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
        self._label = QtGui.QLabel("State")
        self._slider = QtGui.QComboBox(QtCore.Qt.Horizontal)
        self._slider.setMaximumWidth(50)
        self._slider.setMaximum(1)
        on_label = QtGui.QLabel("On")
        on_label.adjustSize()
        off_label = QtGui.QLabel("Off")
        off_label.adjustSize()
        self._layout = QtGui.QGridLayout()
        self._layout.addWidget(self._label, 0, 0, QtCore.Qt.AlignLeft)
        self._layout.addWidget(on_label, 0, 3, QtCore.Qt.AlignRight)
        self._layout.addWidget(off_label, 0, 1, QtCore.Qt.AlignLeft)
        self._layout.addWidget(self._slider, 0, 2)
        self._slider.valueChanged.connect(self._slider_value_changed)
        self.setFixedSize(self.widgetLength, 60)
        self.layout.addLayout(self._layout)
        self._get_state_from_concert()

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


class CameraWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(CameraWidget, self).__init__(name, parent)
        self.object = deviceObject
        layout = QtGui.QGridLayout()
        self.imv = pg.ImageView(self)
        img = self.object.grab()
        self.imv.setImage(img, autoRange=False)
        self.layout.addWidget(self.imv)
        self.timer = QtCore.QTimer(self)
#         self.timer.timeout.connect(self.update)
        self.timer.start(0)
        self._row_number = 3
        self._port_out = PortWidget(self, "port_out")
        self.layout.addWidget(self._port_out)
        self.widget = QtGui.QWidget()
        """_obj_dict is a dictionary where key is a name of widget,
            value[0] is QtGui.QDoubleSpinBox for this parameter
            value[1] is QtGui.QComboBox object with units for this parameter"""
        self._obj_dict = {}
        for param in self.object:
            _parameter_name = param.name
            self._value = str(param.get().result()).split(" ", 1)[0]
            try:
                self._unit = str(param.get().result()).split(" ", 1)[1]
            except:
                pass
            else:
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
        img = self.object.grab()
        now = time()
        self.imv.setImage(img, autoRange=False, autoLevels=False)
        print "fps=", (1.0 / (time() - now))

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

    def close(self):
        self.timer.stop()
        super(CameraWidget, self).close()
        self.destroy()


class FunctionWidget(WidgetPattern):

    def __init__(self, name, parent=None):
        super(FunctionWidget, self).__init__(name, parent)
        self._layout = QtGui.QGridLayout()
        self.func = getattr(processes, name)
        for i in xrange(len(self.func.e_args)):
            if self.func.e_args[i].__class__.__name__ == 'MetaParameterizable':
                exec(
                    "self.%s = PortWidget(self, '%s')" %
                    (str(
                        self.func.f_args[i]),
                        self.func.f_args[i]))
                port = getattr(self, self.func.f_args[i])
                port.setLayoutDirection(QtCore.Qt.RightToLeft)
                self._layout.addWidget(port, i, 1, 1, 2, QtCore.Qt.AlignLeft)

            elif self.func.e_args[i].__class__.__name__ == 'Numeric':
                exec(
                    "self.%s_label = QtGui.QLabel('%s (%s)')" %
                    (str(
                        self.func.f_args[i]), self.func.f_args[i], str(
                        self.func.e_args[i].units or "no unit")))
                label = getattr(self, str(self.func.f_args[i]) + "_label")
                self._layout.addWidget(label, i, 0)
                for j in xrange(self.func.e_args[i].dimension):
                    exec(
                        "self.%s%i = QtGui.QDoubleSpinBox()" %
                        (str(
                            self.func.f_args[i]),
                            j))
                    spin_box = getattr(self, str(self.func.f_args[i]) + str(j))
                    spin_box.setRange(-1000, 1000)
                    spin_box.setValue(0)
                    self._layout.addWidget(spin_box, i, j + 1)
            else:
                print "Sorry, I didn't finish this part yet"

        for j in self.func.e_keywords:
            if self.func.e_keywords[
                    j].__class__.__name__ == 'MetaParameterizable':
                i += 1
                exec("self.%s = PortWidget(self, '%s')" % (str(j), j))
                port = getattr(self, j)
                port.setLayoutDirection(QtCore.Qt.RightToLeft)
                self._layout.addWidget(port, i, 1, 1, 2, QtCore.Qt.AlignLeft)

        for j in self.func.e_keywords:
            i += 1
            class_name = self.func.e_keywords[j].__class__.__name__
            if not (class_name == 'MetaParameterizable' or j == "output"):
                if class_name == 'Numeric':
                    exec(
                        "self.%s_label = QtGui.QLabel('%s (%s)')" %
                        (str(j), j, str(
                            self.func.e_keywords[j].units or "no unit")))
                    label = getattr(self, str(j) + "_label")
                    self._layout.addWidget(label, i, 0)
                    for k in xrange(self.func.e_keywords[j].dimension):
                        exec(
                            "self.%s%i = QtGui.QDoubleSpinBox()" %
                            (str(j), k))
                        spin_box = getattr(self, str(j) + str(k))
                        spin_box.setRange(-1000, 1000)
                        self._layout.addWidget(spin_box, i, 1 + k)
        exec("self.output = PortWidget(self, 'output')")
        port = getattr(self, "output")
        self._layout.addWidget(port, i + 1, 0, QtCore.Qt.AlignLeft)
        self._play_button = QtGui.QToolButton()
        self._play_button.setIcon(
            QtGui.QIcon.fromTheme("media-playback-start"))
        self._play_button.setText("play")
        self._play_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon)
        self._play_button.clicked.connect(self.play_button_clicked)
        self._play_button.adjustSize()
        self._layout.addWidget(
            self._play_button,
            99,
            0,
            1,
            3,
            QtCore.Qt.AlignCenter)
        self.layout.addLayout(self._layout)
        self.adjustSize()
        self.gui = self.parent()

    def play_button_clicked(self):
        args = []
        for i in xrange(len(self.func.e_args)):
            if self.func.e_args[i].__class__.__name__ == 'MetaParameterizable':
                port = getattr(self, self.func.f_args[i])
                try:
                    args.append(getattr(port.get_another_widget(), "object"))
                except:
                    print "%s is not defined!" % port.text()
                    return
            elif self.func.e_args[i].__class__.__name__ == 'Numeric':
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

            if self.func.e_keywords[
                    item].__class__.__name__ == 'MetaParameterizable':
                port = getattr(self, item)
                try:
                    args.append(getattr(port.get_another_widget(), "object"))
                except:
                    break

            elif self.func.e_keywords[item].__class__.__name__ == 'Numeric':
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
            self.func(*args)
        except Exception as e:
            print "Error: ", str(e)
        else:
            print "focus function called"
