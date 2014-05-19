from PyQt4.QtGui import *
from PyQt4.QtCore import *
from concert.devices.lightsources.dummy import LightSource
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.motors.dummy import ContinuousLinearMotor, RotationMotor
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


class WidgetPattern(QGroupBox):

    def __init__(self, name, parent=None):
        super(WidgetPattern, self).__init__(name, parent)
        self.offset = 0
        self.cursor = QCursor
        self.widgetLength = 280
        self.widgetHeight = 80
        global shadowAccepted
        shadowAccepted = False

    def mousePressEvent(self, event):
        self.offset = event.pos()

    def mouseMoveEvent(self, event):
        try:
            self.moveWidget(
                QGroupBox.mapToParent(
                    self,
                    event.pos() -
                    self.offset))
        except:
            QApplication.restoreOverrideCursor()

    def mouseReleaseEvent(self, event):
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
        self.positionUnits = QComboBox()
        self.positionUnits.addItems(["kV", "V", "mV"])
        self.positionUnits.setCurrentIndex(1)

        self.layout = QGridLayout()
        self.layout.addWidget(self.label, 0, 0)
#         self.layout.addWidget(self.value, 0, 1)
        self.layout.addWidget(self.spinValue, 0, 1)
        self.layout.addWidget(self.positionUnits, 0, 2)
        self.setFixedSize(self.widgetLength, 60)
        self.setLayout(self.layout)
        self.positionUnits.currentIndexChanged.connect(self.unitChanged)
        self.spinValue.valueChanged.connect(self.numberChanged)

    def __call__(self):
        return self

    def unitChanged(self, index):
        self.unit = self.positionUnits.currentText()
        if not self.unit == q.get_symbol(str(self.light.intensity.units)):
            new_value = self.light.intensity.ito(
                q.parse_expression(str(self.unit)))
        else:
            new_value = self.light.intensity
        self.spinValue.setValue(float(new_value.magnitude))

    def numberChanged(self):
        num = self.spinValue.text()
        unit = self.positionUnits.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        self.light.intensity = new_value


class MotorWidget(WidgetPattern):

    def __init__(self, name, deviceObject, parent=None):
        super(MotorWidget, self).__init__(name, parent)
        self.motorWidget = deviceObject
        self.unitsDict = {}
        self.unitsDict['meter'] = ["mm", "um"]
        self.unitsDict['degree'] = ["deg", "rad"]
        self.unitsDict['meter / second'] = ["m/s", "mm/s"]
        self.unitsDict['degree / second'] = ["deg/s", 'rad/s']
        self.position = QLabel("Position")
        self.homeButton = QToolButton()
        self.homeButton.setIcon(QIcon.fromTheme("go-home"))
        self.stopButton = QToolButton()
        self.stopButton.setIcon(QIcon.fromTheme("process-stop"))
        self.positionValue = QDoubleSpinBox()
        self.positionValue.setRange(-1000000, 1000000)
        self.positionValue.setAccelerated(True)
        self.positionValue.setDecimals(3)
        self.positionValue.setAlignment(Qt.AlignRight)
        self.positionUnits = QComboBox()
        self.positionUnits.addItems(
            self.unitsDict[str(self.motorWidget.position.units)])
        self.layout = QGridLayout()
        self.layout.addWidget(self.position, 0, 0)
        self.layout.addWidget(self.homeButton, 0, 1)
        self.layout.addWidget(self.stopButton, 0, 2)
        self.layout.addWidget(self.positionValue, 0, 3)
        self.layout.addWidget(self.positionUnits, 0, 4)
        self.setFixedSize(self.widgetLength, 60)
        self.setLayout(self.layout)
        self.positionUnits.currentIndexChanged.connect(
            self.getValueFromConcert)
        self.positionValue.valueChanged.connect(self.positionValueChanged)
        self.homeButton.clicked.connect(self.homeButtonClicked)
        self.stopButton.clicked.connect(self.stopButtonClicked)
        try:
            self.motorWidget.velocity
        except:
            self.velocityExist = False
        else:
            self.velocityExist = True
        if self.velocityExist:
            self.velocity = QLabel("Velocity")
            self.velocityValue = QDoubleSpinBox()
            self.velocityValue.setRange(-1000000, 1000000)
            self.velocityValue.setAccelerated(True)
            self.velocityValue.setDecimals(3)
            self.velocityValue.setAlignment(Qt.AlignRight)
            self.velocityUnits = QComboBox()
            self.velocityUnits.addItems(
                self.unitsDict[str(self.motorWidget.velocity.units)])
            self.layout.addWidget(self.velocity, 1, 0)
            self.layout.addWidget(self.velocityValue, 1, 3)
            self.layout.addWidget(self.velocityUnits, 1, 4)
            self.setFixedSize(self.widgetLength, 80)
            self.velocityUnits.currentIndexChanged.connect(
                self.getValueFromConcert)
            self.velocityValue.valueChanged.connect(self.velocityValueChanged)
        self.getValueFromConcert()

    def __call__(self):
        return self

    def positionValueChanged(self):
        num = self.positionValue.text()
        unit = self.positionUnits.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        try:
            self.motorWidget.position = new_value
        except HardLimitError:
            pass
        self.getValueFromConcert()
        self.checkState()

    def velocityValueChanged(self):
        num = self.velocityValue.text()
        unit = self.velocityUnits.currentText()
        new_value = q.parse_expression(str(num) + str(unit))
        try:
            self.motorWidget.velocity = new_value
        except HardLimitError:
            pass
        self.getValueFromConcert()
        self.checkState()

    def getValueFromConcert(self):
        self.positionValue.valueChanged.disconnect(self.positionValueChanged)
        self.unit = self.positionUnits.currentText()
        if not self.unit == q.get_symbol(str(self.motorWidget.position.units)):
            new_value = self.motorWidget.position.ito(
                q.parse_expression(str(self.unit)))
        else:
            new_value = self.motorWidget.position
        self.positionValue.setValue(float(new_value.magnitude))
        self.positionValue.valueChanged.connect(self.positionValueChanged)
        if self.velocityExist:
            self.velocityValue.valueChanged.disconnect(
                self.velocityValueChanged)
            self.unit1 = str(self.velocityUnits.currentText()).split("/")[0]
            self.unit2 = str(self.velocityUnits.currentText()).split("/")[1]
            self.concertUnit1 = q.get_symbol(
                str(self.motorWidget.velocity.units).split(" / ")[0])
            self.concertUnit2 = q.get_symbol(
                str(self.motorWidget.velocity.units).split(" / ")[1])
            if not self.unit1 == self.concertUnit1 or not self.unit2 == self.concertUnit2:
                new_value = self.motorWidget.velocity.ito(
                    q.parse_expression(str(self.unit1) + "/" + str(self.unit2)))
            else:
                new_value = self.motorWidget.velocity
            self.velocityValue.setValue(float(new_value.magnitude))

            self.velocityValue.valueChanged.connect(self.velocityValueChanged)
        self.checkState()

    def checkState(self):
        state = self.motorWidget.state
        if (state == 'standby') and not (
                self.positionValue.styleSheet() == "background-color: rgb(230, 255, 230);"):
            self.positionValue.setStyleSheet(
                "background-color: rgb(230, 255, 230);")
        elif (state == 'moving') and not (
                self.positionValue.styleSheet() == "background-color: rgb(255, 214, 156);"):
            self.positionValue.setStyleSheet(
                "background-color: rgb(255, 214, 156);")

    def homeButtonClicked(self):
        self.motorWidget.home
        self.getValueFromConcert()

    def stopButtonClicked(self):
        self.motorWidget.stop
        self.checkState()
