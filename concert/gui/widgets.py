from PyQt4.QtGui import *
from PyQt4.QtCore import *
from concert.devices.lightsources.dummy import LightSource
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.motors.dummy import ContinuousLinearMotor, RotationMotor
from concert.devices.positioners.dummy import Positioner
from concert.devices.shutters.dummy import Shutter
from concert.quantities import q
from concert.devices.base import Device
from concert.base import HardLimitError

linear1 = LinearMotor()
linear2 = LinearMotor()
light1 = LightSource()
light2 = LightSource()
contLin3 = ContinuousLinearMotor()
rotation4 = RotationMotor()
contRot5 = ContinuousRotationMotor()
positioner = Positioner()
shutter = Shutter()


class WidgetPattern(QGroupBox):

    """Determines basic device widgets behavior"""

    def __init__(self, name, parent=None):
        super(WidgetPattern, self).__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 24, 0, 0)
        self._offset = 0
        self._cursor = QCursor
        self.widgetLength = 280
        self.widgetHeight = 20
        global shadowAccepted
        shadowAccepted = False
        self.name = QLabel(parent=self)
        self.name.setText(name)
        self.name.adjustSize()
        self.name.move(self.widgetLength / 2 - self.name.width() / 2, 5)
        self.close_button = QToolButton(parent=self)
        self.close_button.resize(24, 24)
        self.close_button.setAutoRaise(True)
        self.close_button.move(
            self.widgetLength -
            self.close_button.width(),
            0)
        self.close_button.setIcon(QIcon.fromTheme("application-exit"))
        self.grid_x_step = 140
        self.grid_y_step = 32
        self._units_dict = {}
        self._units_dict['meter'] = ["mm", "um"]
        self._units_dict['degree'] = ["deg", "rad"]
        self._units_dict['meter / second'] = ["m/s", "mm/s", "um/s"]
        self._units_dict['degree / second'] = ["deg/s", 'rad/s']

    def mousePressEvent(self, event):
        global shadowAccepted
        shadowAccepted = False
        self._offset = event.pos()

    def mouseMoveEvent(self, event):
        if self._offset is not None:
            try:
                self.move_widget(
                    QGroupBox.mapToParent(
                        self,
                        event.pos() -
                        self._offset))
            except:
                QApplication.restoreOverrideCursor()

    def mouseReleaseEvent(self, event):
        global shadowAccepted
        if shadowAccepted:
            self.move_by_grid()
            shadowAccepted = False
        QApplication.restoreOverrideCursor()
        self._offset = None

    def move_widget(self, position):
        global shadowAccepted
        try:
            self.move(position)
        except:
            QApplication.restoreOverrideCursor()
            shadowAccepted = False
        else:
            shadowAccepted = True

    def move_by_grid(self):
        global shadowAccepted
        x, y = self.get_grid_position()
        self.move_widget(QPoint(x, y))
        QApplication.restoreOverrideCursor()
        shadowAccepted = False

    def get_grid_position(self):
        x = self.mapToParent(self.mapFromGlobal(self._cursor.pos())).x()
        y = self.mapToParent(self.mapFromGlobal(self._cursor.pos())).y()
        if x < self.widgetLength:
            x = self.widgetLength
        x = (x / self.grid_x_step * self.grid_x_step) - 130
        y = y / self.grid_y_step * self.grid_y_step
        return x, y

    def get_shadow_status(self):
        global shadowAccepted
        if shadowAccepted:
            return True
        else:
            return False


class LightSourceWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(LightSourceWidget, self).__init__(name, parent)
        self._light = deviceObject
        self._label = QLabel("Intensity")
        self._spin_value = QDoubleSpinBox()
        self._spin_value.setRange(-1000000, 1000000)
        self._spin_value.setDecimals(3)
        self._spin_value.setAccelerated(True)
        self._spin_value.setAlignment(Qt.AlignRight)
        self._intensity_units = QComboBox()
        self._intensity_units.addItems(["kV", "V", "mV"])
        self._intensity_units.setCurrentIndex(1)
        self._layout = QGridLayout()
        self._layout.addWidget(self._label, 0, 0)
        self._layout.addWidget(self._spin_value, 0, 1)
        self._layout.addWidget(self._intensity_units, 0, 2)
        self.setFixedSize(self.widgetLength, 60)
        self.layout.addLayout(self._layout)
        self._intensity_units.currentIndexChanged.connect(self._unit_changed)
        self._spin_value.valueChanged.connect(self._number_changed)

    def __call__(self):
        return self

    def _unit_changed(self, index):
        self._spin_value.valueChanged.disconnect(self._number_changed)
        self._unit = self._intensity_units.currentText()
        if not self._unit == q.get_symbol(str(self._light.intensity.units)):
            _new_value = self._light.intensity.to(q[str(self._unit)])
        else:
            _new_value = self._light.intensity
        self._spin_value.setValue(float(_new_value.magnitude))
        self._spin_value.valueChanged.connect(self._number_changed)

    def _number_changed(self):
        num = self._spin_value.text()
        unit = self._intensity_units.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self._light.intensity = new_value


class MotorWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(MotorWidget, self).__init__(name, parent)
        self._green = "background-color: rgb(230, 255, 230);"
        self._orange = "background-color: rgb(255, 214, 156);"
        self._motor_widget = deviceObject

        self._home_button = QToolButton()
        self._home_button.setIcon(QIcon.fromTheme("go-home"))
        self._stop_button = QToolButton()
        self._stop_button.setIcon(QIcon.fromTheme("process-stop"))
        self._home_button.clicked.connect(self._home_button_clicked)
        self._stop_button.clicked.connect(self._stop_button_clicked)
        self._state = QLabel("state")
        self._state_label = QLabel("")
        self._state_label.setFrameShape(QFrame.WinPanel)
        self._state_label.setFrameShadow(QFrame.Raised)
        self._layout = QGridLayout()
        self._layout.addWidget(self._state, 0, 0)
        self._layout.addWidget(self._state_label, 0, 3, 1, 1, Qt.AlignCenter)
        self._layout.addWidget(self._home_button, 1, 1)
        self._layout.addWidget(self._stop_button, 1, 2)
        self._row_number = 1

        """_obj_dict is a dictionary where key is a name of widget,
            value[0] is QDoubleSpinBox for this parameter
            value[1] is QComboBox object with units for this parameter"""

        self._obj_dict = {}
        for param in self._motor_widget:
            _parameter_name = param.name
            self._value = str(param.get().result()).split(" ", 1)[0]
            self._unit = str(param.get().result()).split(" ", 1)[1]
            parameter_label = QLabel(_parameter_name)
            parameter_value = QDoubleSpinBox()
            parameter_value.setRange(-1000000, 1000000)
            parameter_value.setAccelerated(True)
            parameter_value.setDecimals(3)
            parameter_value.setAlignment(Qt.AlignRight)
            parameter_unit = QComboBox()
            parameter_unit.addItems(
                self._units_dict[str(getattr(self._motor_widget, _parameter_name).units)])
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

    def __call__(self):
        return self

    def _value_changed(self):
        sender = self.sender()
        for key, value in self._obj_dict.iteritems():
            if value[0] == sender:
                num = sender.text()
                unit = value[1].currentText()
                new_value = q.parse_expression(str(num) + str(unit))
                try:
                    setattr(self._motor_widget, key, new_value)
                except HardLimitError:
                    pass
                self._get_value_from_concert()
                self._check_state()

    def _get_value_from_concert(self):
        for _key in self._obj_dict.keys():
            parameter_value = self._obj_dict[_key][0]
            parameter_unit = self._obj_dict[_key][1]
            parameter_value.valueChanged.disconnect(self._value_changed)
            self._unit = parameter_unit.currentText()
            if not q[str(self._unit)] == q[str(getattr(self._motor_widget, _key).units)]:
                new_value = getattr(self._motor_widget, _key).to(
                    q[str(self._unit)])
            else:
                new_value = getattr(self._motor_widget, _key)
            parameter_value.setValue(float(new_value.magnitude))
            parameter_value.valueChanged.connect(self._value_changed)
        self._check_state()

    def _check_state(self):
        state = self._motor_widget.state
        if (state == 'standby'):
            self._state_label.setStyleSheet(self._green)
            self._state_label.setText("standby")
        elif (state == 'moving'):
            self._state_label.setStyleSheet(self._orange)
            self._state_label.setText("moving")

    def _home_button_clicked(self):
        self._motor_widget.home
        self._get_value_from_concert()

    def _stop_button_clicked(self):
        self._motor_widget.stop
        self._check_state()


class PositionerWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(PositionerWidget, self).__init__(name, parent)
        self._positioner_widget = deviceObject
        self._button_left = QToolButton()
        self._button_left.setIcon(QIcon.fromTheme("go-previous"))
        self._button_left.setAutoRaise(True)
        self._button_right = QToolButton()
        self._button_right.setIcon(QIcon.fromTheme("go-next"))
        self._button_right.setAutoRaise(True)
        self._button_up = QToolButton()
        self._button_up.setIcon(QIcon.fromTheme("go-up"))
        self._button_up.setAutoRaise(True)
        self._button_down = QToolButton()
        self._button_down.setIcon(QIcon.fromTheme("go-down"))
        self._button_down.setAutoRaise(True)
        self._button_back = QToolButton()
        self._button_back.setIcon(QIcon.fromTheme("go-down"))
        self._button_back.setAutoRaise(True)
        self._button_forward = QToolButton()
        self._button_forward.setIcon(QIcon.fromTheme("go-up"))
        self._button_forward.setAutoRaise(True)
        self._button_clockwise = QToolButton()
        self._button_clockwise.setIcon(QIcon.fromTheme("object-rotate-right"))
        self._button_clockwise.setAutoRaise(True)
        self._button_counterclockwise = QToolButton()
        self._button_counterclockwise.setIcon(
            QIcon.fromTheme("object-rotate-left"))
        self._button_counterclockwise.setAutoRaise(True)
        xy_label = QLabel("x  y")
        z_label = QLabel("z")
        self._rotation_axis = QComboBox()
        self._rotation_axis.addItems(["x", "y", "z"])
        self._rotation_axis.setMaximumWidth(50)
        self._rotation_axis.setMinimumWidth(35)
        self._step_value = QDoubleSpinBox()
        self._step_value.setRange(0, 1000000)
        self._step_value.setAccelerated(True)
        self._step_value.setDecimals(2)
        self._step_value.setAlignment(Qt.AlignRight)
        self._step_unit = QComboBox()
        self._step_unit.addItems(self._units_dict["meter"])
        self._rotation_step_value = QDoubleSpinBox()
        self._rotation_step_value.setRange(0, 1000000)
        self._rotation_step_value.setAccelerated(True)
        self._rotation_step_value.setDecimals(2)
        self._rotation_step_value.setAlignment(Qt.AlignRight)
        self._rotation_step_unit = QComboBox()
        self._rotation_step_unit.addItems(self._units_dict["degree"])
        line = QFrame()
        line.setGeometry(QRect(320, 150, 118, 3))
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        line1 = QFrame()
        line1.setGeometry(QRect(320, 150, 118, 3))
        line1.setFrameShape(QFrame.VLine)
        line1.setFrameShadow(QFrame.Sunken)
        self._layout = QGridLayout()
        self._layout.addWidget(self._button_left, 1, 0)
        self._layout.addWidget(self._button_right, 1, 2)
        self._layout.addWidget(self._button_forward, 0, 1)
        self._layout.addWidget(self._button_back, 2, 1)
        self._layout.addWidget(self._button_up, 0, 4, Qt.AlignBottom)
        self._layout.addWidget(self._button_down, 2, 4, Qt.AlignTop)
        self._layout.addWidget(self._button_clockwise, 1, 10, Qt.AlignLeft)
        self._layout.addWidget(
            self._button_counterclockwise,
            1,
            7,
            Qt.AlignRight)
        self._layout.addWidget(self._step_value, 4, 0, 1, 3)
        self._layout.addWidget(self._step_unit, 4, 3, 1, 2)
        self._layout.addWidget(self._rotation_step_value, 4, 7, 1, 2)
        self._layout.addWidget(self._rotation_step_unit, 4, 9, 1, 2)
        self._layout.addWidget(xy_label, 1, 1)
        self._layout.addWidget(z_label, 1, 4, Qt.AlignCenter)
        self._layout.addWidget(self._rotation_axis, 1, 8, 1, 2, Qt.AlignCenter)
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

    def __call__(self):
        return self

    def _button_left_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self._positioner_widget.left(new_value)

    def _button_right_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self._positioner_widget.right(new_value)

    def _button_up_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self._positioner_widget.up(new_value)

    def _button_down_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self._positioner_widget.down(new_value)

    def _button_forward_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self._positioner_widget.forward(new_value)

    def _button_back_clicked(self):
        num = self._step_value.text()
        unit = self._step_unit.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self._positioner_widget.back(new_value)

    def _button_clockwise_clicked(self):
        pass

    def _button_counterclockwise_clicked(self):
        pass


class ShutterWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(ShutterWidget, self).__init__(name, parent)
        self._shutter = deviceObject
        self._label = QLabel("State")
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMaximumWidth(50)
        self._slider.setMaximum(1)
        on_label = QLabel("On")
        on_label.adjustSize()
        off_label = QLabel("Off")
        off_label.adjustSize()
        self._layout = QGridLayout()
        self._layout.addWidget(self._label, 0, 0, Qt.AlignLeft)
        self._layout.addWidget(on_label, 0, 3, Qt.AlignRight)
        self._layout.addWidget(off_label, 0, 1, Qt.AlignLeft)
        self._layout.addWidget(self._slider, 0, 2)
        self._slider.valueChanged.connect(self._slider_value_changed)
        self.setFixedSize(self.widgetLength, 60)
        self.layout.addLayout(self._layout)

    def __call__(self):
        return self

    def _slider_value_changed(self):
        value = self._slider.value()
        if value == 1:
            self._shutter.open()
        elif value == 0:
            self._shutter.close()
