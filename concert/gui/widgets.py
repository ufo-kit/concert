from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
import os
from concert.devices.lightsources.dummy import LightSource
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.motors.dummy import ContinuousLinearMotor, RotationMotor
from concert.quantities import q
from concert.devices.base import Device
from concert.base import HardLimitError

# scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
# os.chdir(scriptPath)
# sys.path.append(os.path.realpath("../../../../.local/share/concert/"))
# import tutorial
# import gc

linear1 = LinearMotor()
linear2 = LinearMotor()
light1 = LightSource()
light2 = LightSource()
contLin3 = ContinuousLinearMotor()
rotation4 = RotationMotor()
contRot5 = ContinuousRotationMotor()


class WidgetPattern(QGroupBox):

    """Determines basic device widgets behavior"""

    def __init__(self, name, parent=None):
        super(WidgetPattern, self).__init__(name, parent)
        self._offset = 0
        self._cursor = QCursor
        self.widgetLength = 280
        self.widgetHeight = 20
        global shadowAccepted
        shadowAccepted = False

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
        _x = _x / self.widgetLength * self.widgetLength
        _y = _y / self.widgetHeight * self.widgetHeight
        return _x, _y

    def get_shadow_status(self):
        global shadowAccepted
        if shadowAccepted:
            return True
        else:
            return False


class LightSourceWidget(WidgetPattern):

    def __init__(self, name, parent=None):
        super(LightSourceWidget, self).__init__(name, parent)
        self._light = LightSource()
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
        self.setLayout(self._layout)
        self._intensity_units.currentIndexChanged.connect(self._unit_changed)
        self._spin_value.valueChanged.connect(self._number_changed)

    def __call__(self):
        return self

    def _unit_changed(self, index):
        self._unit = self._intensity_units.currentText()
        if not self._unit == q.get_symbol(str(self._light.intensity.units)):
            _new_value = self._light.intensity.ito(
                q.parse_expression(str(self._unit)))
        else:
            _new_value = self._light.intensity
        self._spin_value.setValue(float(_new_value.magnitude))

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
        self._units_dict['meter / second'] = ["m/s", "mm/s"]
        self._units_dict['degree / second'] = ["deg/s", 'rad/s']
        self._state = QLabel("State")
        self._state_label = QLabel("")
        self._state_label.setFrameShape(QFrame.WinPanel)
        self._state_label.setFrameShadow(QFrame.Raised)
        self._position = QLabel("Position")
        self._home_button = QToolButton()
        self._home_button.setIcon(QIcon.fromTheme("go-home"))
        self._stop_button = QToolButton()
        self._stop_button.setIcon(QIcon.fromTheme("process-stop"))
        self._position_value = QDoubleSpinBox()
        self._position_value.setRange(-1000000, 1000000)
        self._position_value.setAccelerated(True)
        self._position_value.setDecimals(3)
        self._position_value.setAlignment(Qt.AlignRight)
        self._positionUnits = QComboBox()
        self._positionUnits.addItems(
            self._units_dict[str(self._motor_widget.position.units)])
        self._layout = QGridLayout()
        self._layout.addWidget(self._state, 0, 0)
        self._layout.addWidget(self._state_label, 0, 3, 1, 1, Qt.AlignCenter)
        self._layout.addWidget(self._position, 1, 0)
        self._layout.addWidget(self._home_button, 1, 1)
        self._layout.addWidget(self._stop_button, 1, 2)
        self._layout.addWidget(self._position_value, 1, 3)
        self._layout.addWidget(self._positionUnits, 1, 4)
        self.setFixedSize(self.widgetLength, 80)
        self.setLayout(self._layout)
        self._positionUnits.currentIndexChanged.connect(
            self._get_value_from_concert)
        self._position_value.valueChanged.connect(self._position_value_changed)
        self._home_button.clicked.connect(self._home_button_clicked)
        self._stop_button.clicked.connect(self._stop_button_clicked)
        try:
            self._motor_widget.velocity
        except:
            self._velocity_exist = False
        else:
            self._velocity_exist = True
        if self._velocity_exist:
            self._velocity = QLabel("Velocity")
            self._velocity_value = QDoubleSpinBox()
            self._velocity_value.setRange(-1000000, 1000000)
            self._velocity_value.setAccelerated(True)
            self._velocity_value.setDecimals(3)
            self._velocity_value.setAlignment(Qt.AlignRight)
            self._velocity_units = QComboBox()
            self._velocity_units.addItems(
                self._units_dict[str(self._motor_widget.velocity.units)])
            self._layout.addWidget(self._velocity, 2, 0)
            self._layout.addWidget(self._velocity_value, 2, 3)
            self._layout.addWidget(self._velocity_units, 2, 4)
            self.setFixedSize(self.widgetLength, 100)
            self._velocity_units.currentIndexChanged.connect(
                self._get_value_from_concert)
            self._velocity_value.valueChanged.connect(
                self._velocity_value_changed)
        self._get_value_from_concert()

    def __call__(self):
        return self

    def _position_value_changed(self):
        _num = self._position_value.text()
        _unit = self._positionUnits.currentText()
        _new_value = q.parse_expression(str(_num) + str(_unit))
        try:
            self._motor_widget.position = _new_value
        except HardLimitError:
            pass
        self._get_value_from_concert()
        self._check_state()

    def _velocity_value_changed(self):
        _num = self._velocity_value.text()
        _unit = self._velocity_units.currentText()
        _new_value = q.parse_expression(str(_num) + str(_unit))
        try:
            self._motor_widget.velocity = _new_value
        except HardLimitError:
            pass
        self._get_value_from_concert()
        self._check_state()

    def _get_value_from_concert(self):
        self._position_value.valueChanged.disconnect(
            self._position_value_changed)
        self._unit = self._positionUnits.currentText()
        if not self._unit == q.get_symbol(str(self._motor_widget.position.units)):
            _new_value = self._motor_widget.position.ito(
                q.parse_expression(str(self._unit)))
        else:
            _new_value = self._motor_widget.position
        self._position_value.setValue(float(_new_value.magnitude))
        self._position_value.valueChanged.connect(self._position_value_changed)
        if self._velocity_exist:
            self._velocity_value.valueChanged.disconnect(
                self._velocity_value_changed)
            self._unit1 = str(self._velocity_units.currentText()).split("/")[0]
            self._unit2 = str(self._velocity_units.currentText()).split("/")[1]
            self.concertUnit1 = q.get_symbol(
                str(self._motor_widget.velocity.units).split(" / ")[0])
            self.concertUnit2 = q.get_symbol(
                str(self._motor_widget.velocity.units).split(" / ")[1])
            if not self._unit1 == self.concertUnit1 or not self._unit2 == self.concertUnit2:
                _new_value = self._motor_widget.velocity.ito(
                    q.parse_expression(str(self._unit1) + "/" + str(self._unit2)))
            else:
                _new_value = self._motor_widget.velocity
            self._velocity_value.setValue(float(_new_value.magnitude))

            self._velocity_value.valueChanged.connect(
                self._velocity_value_changed)
        self._check_state()

    def _check_state(self):
        _state = self._motor_widget.state
        if (_state == 'standby') and not (self._state_label.styleSheet() == self._green):
            self._state_label.setStyleSheet(self._green)
            self._state_label.setText("standby")
        elif (_state == 'moving') and not (self._state_label.styleSheet() == self._orange):
            self._state_label.setStyleSheet(self._orange)
            self._state_label.setText("moving")

    def _home_button_clicked(self):
        self._motor_widget.home
        self._get_value_from_concert()

    def _stop_button_clicked(self):
        self._motor_widget.stop
        self._check_state()
