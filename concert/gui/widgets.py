import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from concert.devices.lightsources.dummy import LightSource
from concert.quantities import q


class MyGroupBox(QGroupBox):

    def __init__(self, name, parent=None):
        super(MyGroupBox, self).__init__(name, parent)
        self.offset = 0
        self.cursor = QCursor
        self.widgetLength = 280
        self.widgetHeight = 100
        global shadowAccepted
        shadowAccepted = False

    def mousePressEvent(self, event):

        super(MyGroupBox, self).mousePressEvent(event)
        self.offset = event.pos()

        QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        super(MyGroupBox, self).mouseMoveEvent(event)
        try:

            self.moveWidget(
                QGroupBox.mapToParent(
                    self,
                    event.pos() -
                    self.offset))

        except:

            QApplication.restoreOverrideCursor()


#
    def mouseReleaseEvent(self, event):
        super(MyGroupBox, self).mouseReleaseEvent(event)

        if shadowAccepted:

            self.moveFollowGrid()

        QApplication.restoreOverrideCursor()

    def moveWidget(self, position):
        #         if not QApplication.overrideCursor() is None:
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


class MyDoubleSpinBox(QDoubleSpinBox):

    def __init__(self):
        super(MyDoubleSpinBox, self).__init__()

    def mousePressEvent(self, event):
        super(MyDoubleSpinBox, self).mousePressEvent(event)
        QApplication.restoreOverrideCursor()

    def mouseReleaseEvent(self, event):
        super(MyDoubleSpinBox, self).mouseReleaseEvent(event)
        QApplication.restoreOverrideCursor()


class LightSourceWidget(MyGroupBox):

    def __init__(self, name, parent=None):
        super(LightSourceWidget, self).__init__(name, parent)

        self.light = LightSource()

        self.applyButton = QPushButton("Apply")
        self.applyButton.setMaximumSize(81, 22)
        self.label = QLabel("Intensity")
        self.value = QLabel("")
        self.value.setStyleSheet("background-color: rgb(210, 255, 193)")
        self.value.setFrameShape(QFrame.WinPanel)
        self.value.setFrameShadow(QFrame.Sunken)
        self.spinValue = MyDoubleSpinBox()
#         self.spinValue.minimum(-1000)
        self.spinValue.setRange(-100000, 100000)

        self.units = QComboBox()
        self.units.addItems(["kV", "V", "mV"])
        self.units.setCurrentIndex(1)
        self.previousIndex = self.units.currentIndex()
#         self.units.setMaximumSize(41, 22)

        self.layout = QGridLayout()
        self.layout.addWidget(self.label, 0, 0)
        self.layout.addWidget(self.value, 0, 1)
        self.layout.addWidget(self.spinValue, 0, 2)
        self.layout.addWidget(self.units, 0, 3)
#         self.layout.addWidget(self.applyButton, 1, 2, 1, 2,Qt.AlignRight)
        self.setFixedSize(self.widgetLength, 100)
        self.setLayout(self.layout)

        self.units.currentIndexChanged.connect(self.unitChanged)
        self.spinValue.valueChanged.connect(self.numberChanged)
        self.update()

    def __call__(self):
        return self

    def unitChanged(self, index):
        number = self.spinValue.text()
        dif = self.units.currentIndex() - self.previousIndex
        number = float(number) * (10 ** (3 * dif))
        self.spinValue.setValue(number)
        self.previousIndex = self.units.currentIndex()

    def numberChanged(self):

        num = self.spinValue.text()
        unit = self.units.currentText()

        new_value = q.parse_expression(str(num) + str(unit))

#         for param in device:
#             print param.name, param.get().result()

        self.light.intensity = new_value
        self.update()

    def update(self):
        self.value.setText(str(self.light.intensity))
