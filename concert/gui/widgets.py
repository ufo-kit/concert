import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from concert.devices.lightsources.dummy import LightSource
from concert.devices.motors.dummy import LinearMotor
from concert.quantities import q
from concert.base import HardLimitError


class WidgetPattern(QGroupBox):

    def __init__(self, name, parent=None):
        super(WidgetPattern, self).__init__(name, parent)
        self.offset = 0
        self.cursor = QCursor
        self.widgetLength = 280
        self.widgetHeight = 80
        global shadowAccepted
        shadowAccepted = False

    def mouseDoubleClickEvent(self, event):
        super(WidgetPattern, self).mousePressEvent(event)
        self.offset = event.pos()

    def mouseMoveEvent(self, event):
        super(WidgetPattern, self).mouseMoveEvent(event)
        try:
            self.moveWidget(
                QGroupBox.mapToParent(
                    self,
                    event.pos() -
                    self.offset))
        except:
            QApplication.restoreOverrideCursor()

    def mouseReleaseEvent(self, event):
        super(WidgetPattern, self).mouseReleaseEvent(event)
        if shadowAccepted:
            self.moveFollowGrid()
        QApplication.restoreOverrideCursor()

    def moveWidget(self, position):
        global shadowAccepted
        try:
            self.move(position)
            shadowAccepted = True
        except:
            QApplication.restoreOverrideCursor()
            shadowAccepted = False

    def moveFollowGrid(self):
        global shadowAccepted
        x, y = self.getGridPosition()
        self.moveWidget(QPoint(x, y))
        shadowAccepted = False
        QApplication.restoreOverrideCursor()

    def getGridPosition(self):
        x = self.mapToParent(self.mapFromGlobal(self.cursor.pos())).x()
        y = self.mapToParent(self.mapFromGlobal(self.cursor.pos())).y()
        if x < self.widgetLength:
            x = self.widgetLength
        x = x / self.widgetLength * self.widgetLength
        y = y / self.widgetHeight * self.widgetHeight
        return x, y

    def getShadowStatus(self):
        global shadowAccepted
        if shadowAccepted:
            return True
        else:
            return False


class LightSourceWidget(WidgetPattern):

    def __init__(self, name, parent=None):
        super(LightSourceWidget, self).__init__(name, parent)
        self.light = LightSource()
        self.label = QLabel("Intensity")
        self.spinValue = QDoubleSpinBox()
        self.spinValue.setRange(-1000000, 1000000)
        self.spinValue.setDecimals(3)
        self.spinValue.setAccelerated(True)
        self.spinValue.setAlignment(Qt.AlignRight)
        self.units = QComboBox()
        self.units.addItems(["kV", "V", "mV"])
        self.units.setCurrentIndex(1)
        self.previousIndex = self.units.currentIndex()
        self.layout = QGridLayout()
        self.layout.addWidget(self.label, 0, 0)
#         self.layout.addWidget(self.value, 0, 1)
        self.layout.addWidget(self.spinValue, 0, 1)
        self.layout.addWidget(self.units, 0, 2)
        self.setFixedSize(self.widgetLength, 60)
        self.setLayout(self.layout)
        self.units.currentIndexChanged.connect(self.unitChanged)
        self.spinValue.valueChanged.connect(self.numberChanged)

    def __call__(self):
        return self

    def unitChanged(self, index):
        self.unit = self.units.currentText()
        if not self.unit == q.get_symbol(str(self.light.intensity.units)):
            new_value = self.light.intensity.ito(
                q.parse_expression(str(self.unit)))
        else:
            new_value = self.light.intensity
        self.spinValue.setValue(float(new_value.magnitude))

    def numberChanged(self):
        num = self.spinValue.text()
        unit = self.units.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.light.intensity = new_value


class MotorWidget(WidgetPattern):

    def __init__(self, name, parent=None):
        super(MotorWidget, self).__init__(name, parent)
        self.linearMotor = LinearMotor()
        self.label = QLabel("Position")
        self.homeButton = QToolButton()
        self.homeButton.setIcon(QIcon("media/home.png"))
        self.stopButton = QToolButton()
        self.stopButton.setIcon(QIcon("media/stop.png"))
        self.spinValue = QDoubleSpinBox()
        self.spinValue.setRange(-1000000, 1000000)
        self.spinValue.setAccelerated(True)
        self.spinValue.setDecimals(3)
        self.spinValue.setAlignment(Qt.AlignRight)
        self.units = QComboBox()
        self.units.addItems(["m", "mm", "um"])
        self.units.setCurrentIndex(1)
        self.previousIndex = self.units.currentIndex()
        self.layout = QGridLayout()
        self.layout.addWidget(self.label, 0, 0)
        self.layout.addWidget(self.homeButton, 0, 1)
        self.layout.addWidget(self.stopButton, 0, 2)
        self.layout.addWidget(self.spinValue, 0, 3)
        self.layout.addWidget(self.units, 0, 4)
        self.setFixedSize(self.widgetLength, 60)
        self.setLayout(self.layout)
        self.units.currentIndexChanged.connect(self.getValueFromConcert)
        self.spinValue.valueChanged.connect(self.numberChanged)
        self.homeButton.clicked.connect(self.homeButtonClicked)
        self.stopButton.clicked.connect(self.stopButtonClicked)
        self.getValueFromConcert()

    def __call__(self):
        return self

    def numberChanged(self):
        self.num = self.spinValue.text()
        self.unit = self.units.currentText()
        new_value = q.parse_expression(str(self.num) + str(self.unit))
        try:
            self.linearMotor.position = new_value
        except HardLimitError:
            self.getValueFromConcert()
        self.checkState()

    def getValueFromConcert(self):
        self.spinValue.valueChanged.disconnect(self.numberChanged)
        self.unit = self.units.currentText()
        if not self.unit == q.get_symbol(str(self.linearMotor.position.units)):
            new_value = self.linearMotor.position.ito(
                q.parse_expression(str(self.unit)))
        else:
            new_value = self.linearMotor.position
        self.spinValue.setValue(float(new_value.magnitude))
        self.checkState()
        self.spinValue.valueChanged.connect(self.numberChanged)

    def checkState(self):
        state = self.linearMotor.state
        if (state == 'standby') and not (
                self.spinValue.styleSheet() == "background-color: rgb(230, 255, 230);"):
            self.spinValue.setStyleSheet(
                "background-color: rgb(230, 255, 230);")
        elif (state == 'moving') and not (
                self.spinValue.styleSheet() == "background-color: rgb(255, 214, 156)"):
            self.spinValue.setStyleSheet(
                "background-color: rgb(255, 214, 156);")

    def homeButtonClicked(self):
        self.linearMotor.home
        self.getValueFromConcert()

    def stopButtonClicked(self):
        self.linearMotor.stop
        self.checkState()
