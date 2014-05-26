from PyQt4.QtGui import *
from PyQt4.QtCore import *
from concert.devices.lightsources.dummy import LightSource
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.motors.dummy import ContinuousLinearMotor, RotationMotor
from concert.devices.positioners.dummy import Positioner
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
        self.close_button.move(self.widgetLength - self.close_button.width(), 0)
        self.close_button.setIcon(QIcon.fromTheme("application-exit"))
        self.grid_x_step = 140
        self.grid_y_step = 32

    def mousePressEvent(self, event):
        self._offset = event.pos()

    def mouseMoveEvent(self, event):
        try:
            self.move_widget(
                QGroupBox.mapToParent(
                    self,
                    event.pos() -
                    self._offset))
        except:
            QApplication.restoreOverrideCursor()

    def mouseReleaseEvent(self, event):
        if shadowAccepted:
            self.move_by_grid()
        QApplication.restoreOverrideCursor()

    def move_widget(self, position):
        global shadowAccepted
        try:
            self.move(position)
            shadowAccepted = True
        except:
            QApplication.restoreOverrideCursor()
            shadowAccepted = False

    def move_by_grid(self):
        global shadowAccepted
        x, y = self.get_grid_position()
        self.move_widget(QPoint(x, y))
        shadowAccepted = False
        QApplication.restoreOverrideCursor()

    def get_grid_position(self):
        _x = self.mapToParent(self.mapFromGlobal(self._cursor.pos())).x()
        _y = self.mapToParent(self.mapFromGlobal(self._cursor.pos())).y()
        if _x < self.widgetLength:
            _x = self.widgetLength
        _x = (_x / self.grid_x_step * self.grid_x_step) - 130
        _y = _y / self.grid_y_step * self.grid_y_step
        return _x, _y

    def get_shadow_status(self):
        global shadowAccepted
        if shadowAccepted:
            return True
        else:
            return False


class LightSourceWidget(WidgetPattern):

    def __init__(self, name, deviceObject,parent=None):
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
        _num = self._spin_value.text()
        _unit = self._intensity_units.currentText()
        _new_value = q.parse_expression(str(_num) + str(_unit))
        self._light.intensity = _new_value


class MotorWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(MotorWidget, self).__init__(name, parent)
        self._green = "background-color: rgb(230, 255, 230);"
        self._orange = "background-color: rgb(255, 214, 156);"
        self._motor_widget = deviceObject
        self._units_dict = {}
        self._units_dict['meter'] = ["mm", "um"]
        self._units_dict['degree'] = ["deg", "rad"]
        self._units_dict['meter / second'] = ["m/s", "mm/s", "um/s"]
        self._units_dict['degree / second'] = ["deg/s", 'rad/s']
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
            _parameter_label = QLabel(_parameter_name)
            _parameter_value = QDoubleSpinBox()
            _parameter_value.setRange(-1000000, 1000000)
            _parameter_value.setAccelerated(True)
            _parameter_value.setDecimals(3)
            _parameter_value.setAlignment(Qt.AlignRight)
            _parameter_unit = QComboBox()
            _parameter_unit.addItems(
                self._units_dict[str(getattr(self._motor_widget, _parameter_name).units)])
            _parameter_unit.setObjectName(_parameter_name)
            self._obj_dict[_parameter_name] = [
                _parameter_value,
                _parameter_unit]
            self._layout.addWidget(_parameter_label, self._row_number, 0)
            self._layout.addWidget(_parameter_value, self._row_number, 3)
            self._layout.addWidget(_parameter_unit, self._row_number, 4)
            _parameter_value.valueChanged.connect(self._value_changed)
            _parameter_unit.currentIndexChanged.connect(
                self._get_value_from_concert)
            self._row_number += 1
        self.setFixedSize(self.widgetLength, 60 + (self._row_number - 1) * 25)
        self.layout.addLayout(self._layout)
        self._get_value_from_concert()

    def __call__(self):
        return self

    def _value_changed(self):
        _sender = self.sender()
        for _key, _value in self._obj_dict.iteritems():
            if _value[0] == _sender:
                _num = _sender.text()
                _unit = _value[1].currentText()
                _new_value = q.parse_expression(str(_num) + str(_unit))
                try:
                    setattr(self._motor_widget, _key, _new_value)
                except HardLimitError:
                    pass
                self._get_value_from_concert()
                self._check_state()

    def _get_value_from_concert(self):
        for _key in self._obj_dict.keys():
            _parameter_value = self._obj_dict[_key][0]
            _parameter_unit = self._obj_dict[_key][1]
            _parameter_value.valueChanged.disconnect(self._value_changed)
            self._unit = _parameter_unit.currentText()
            if not q[str(self._unit)] == q[str(getattr(self._motor_widget, _key).units)]:
                _new_value = getattr(self._motor_widget, _key).to(
                    q[str(self._unit)])
            else:
                _new_value = getattr(self._motor_widget, _key)
            _parameter_value.setValue(float(_new_value.magnitude))
            _parameter_value.valueChanged.connect(self._value_changed)
        self._check_state()

    def _check_state(self):
        _state = self._motor_widget.state
        if (_state == 'standby'):
            self._state_label.setStyleSheet(self._green)
            self._state_label.setText("standby")
        elif (_state == 'moving'):
            self._state_label.setStyleSheet(self._orange)
            self._state_label.setText("moving")

    def _home_button_clicked(self):
        self._motor_widget.home
        self._get_value_from_concert()

    def _stop_button_clicked(self):
        self._motor_widget.stop
        self._check_state()


class PositionerWidget(WidgetPattern):

    def __init__(self, name, deviceObject,parent=None):
        super(PositionerWidget, self).__init__(name, parent)
        self._positioner_widget = deviceObject
        self._button_left = QToolButton()
        self._button_left.setIcon(QIcon.fromTheme("go-previous"))
        self._button_right = QToolButton()
        self._button_right.setIcon(QIcon.fromTheme("go-next"))
        self._button_up = QToolButton()
        self._button_up.setIcon(QIcon.fromTheme("go-up"))
        self._button_down = QToolButton()
        self._button_down.setIcon(QIcon.fromTheme("go-down"))
        self._vertical_axis = QComboBox()
        self._vertical_axis.addItems(["x", "y", "z"])
        self._horizontal_axis = QComboBox()
        self._horizontal_axis.addItems(["x", "y", "z"])
        self._layout = QGridLayout()
        self._layout.addWidget(self._button_left, 1, 0, Qt.AlignRight)
        self._layout.addWidget(self._button_right, 1, 2, Qt.AlignLeft)
        self._layout.addWidget(self._button_up, 0, 1, Qt.AlignBottom)
        self._layout.addWidget(self._button_down, 2, 1, Qt.AlignTop)
        self._layout.addWidget(self._vertical_axis, 0, 2, Qt.AlignLeft)
        self._layout.addWidget(self._horizontal_axis, 1, 2, Qt.AlignCenter)
        self.layout.addLayout(self._layout)
        self.setFixedSize(self.widgetLength, 100)
        
        self._button_left.clicked.connect(self._button_left_clicked)
        self._button_right.clicked.connect(self._button_right_clicked)
        self._button_up.clicked.connect(self._button_up_clicked)
        self._button_down.clicked.connect(self._button_down_clicked)
        
    def __call__(self):
        return self
    
    def _button_left_clicked(self):
        _index=self._horizontal_axis.currentIndex()
    
    def _button_right_clicked(self):
        print "right"
    
    def _button_up_clicked(self):
        print "up"
    
    def _button_down_clicked(self):
        print "down"
    

