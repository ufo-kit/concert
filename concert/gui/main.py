'''
Created on Apr 28, 2014

@author: Pavel Rybalko (ANKA)
'''

import sys
import os

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from widgets import MyGroupBox, LightSourceWidget


class ConcertGUI(QWidget):

    def __init__(self, parent=None):
        super(ConcertGUI, self).__init__(parent)

        self.cursor = QCursor

        self.deviceList = MyTreeWidget()
        self.deviceList.setFixedWidth(150)
        self.deviceList.header().setStretchLastSection(False)
        self.deviceList.setHeaderItem(QTreeWidgetItem(["Devices"]))
        motors = QTreeWidgetItem(self.deviceList, ["Motors"])

        for x in ["Linear", "ContinuousLinear", "Rotation"]:
            QTreeWidgetItem(motors, [x])
        motors.setExpanded(True)

        QTreeWidgetItem(self.deviceList, ["Cameras"])
        QTreeWidgetItem(self.deviceList, ["LightSourse"])
        self.deviceList.setColumnWidth(0, 140)

        self.mainLayout = QHBoxLayout()
        self.fieldLayout = QGridLayout()
        self.mainLayout.addWidget(self.deviceList, 0, Qt.AlignLeft)

        self.mainLayout.addLayout(self.fieldLayout)

        self.setLayout(self.mainLayout)
        self.setWindowTitle("Concert GUI")

        self.resize(1024, 786)

        self.widget = MyGroupBox("")

#         self.deviceList.itemPressed.connect(self.itemPressedMethod)

    def createLinear(self):

        self.button = QPushButton("Linear")
        self.label = QLabel("Linear")
        self.line = QLineEdit()
        self.widget = MyGroupBox("LinearMotor", self)
        self.motorvbox = QVBoxLayout()
        self.motorvbox.addWidget(self.label)
        self.motorvbox.addWidget(self.line)
        self.motorvbox.addWidget(self.button)
        self.widget.setLayout(self.motorvbox)
        self.widget.setFixedSize(200, 100)
        self.setObjectName('widget%d')
#         self.fieldLayout.addWidget(self.widget)
        self.widget.show()

    def createContinuousLinear(self):

        self.button = QPushButton("ContinuousLinear")
        self.button.setStyleSheet("background-color: yellow")
        self.label = QLabel("ContinuousLinear")
        self.line = QLineEdit()
        self.widget = MyGroupBox("ContinuousLinear", self)
        self.motorvbox = QVBoxLayout()
        self.motorvbox.addWidget(self.label)
        self.motorvbox.addWidget(self.line)
        self.motorvbox.addWidget(self.button)
        self.widget.setLayout(self.motorvbox)
        self.widget.setFixedSize(200, 100)
        self.setObjectName('ConctinuousWidget%d')
        self.widget.show()

    def createRotation(self):

        self.button = QPushButton("Rotation")
        self.button.setStyleSheet("background-color: red")
        self.label = QLabel("Rotation")
        self.line = QLineEdit()
        self.widget = MyGroupBox("Rotation", self)

        self.motorvbox = QVBoxLayout()
        self.motorvbox.addWidget(self.label)
        self.motorvbox.addWidget(self.line)
        self.motorvbox.addWidget(self.button)
        self.widget.setLayout(self.motorvbox)
        self.widget.setFixedSize(200, 100)
        self.widget.setObjectName('RotationWidget%d')
        self.widget.show()

    def createCameras(self):

        self.button = QPushButton("Camera")
        self.button.setStyleSheet("background-color: green")
        self.label = QLabel("Camera")
        self.line = QLineEdit()
        self.widget = MyGroupBox("Camera", self)
        self.motorvbox = QVBoxLayout()
        self.motorvbox.addWidget(self.label)
        self.motorvbox.addWidget(self.line)
        self.motorvbox.addWidget(self.button)
        self.widget.setLayout(self.motorvbox)
        self.widget.setFixedSize(200, 100)
        self.setObjectName('CameraWidget%d')
        self.widget.show()

    def createLightSourse(self):
        global count

        self.widget = LightSourceWidget("LightSourse", self)

        self.widget.setObjectName('LightSource')

        self.widget().show()

    def paintEvent(self, event):

        self.widgetLength, self.widgetHeight = self.widget.widgetLength, self.widget.widgetHeight
        if self.widget.getShadowStatus():
            qp = QPainter()
            qp.begin(self)
            qp.setBrush(QColor("#ffcccc"))
            qp.setPen(Qt.NoPen)
            self.x, self.y = self.widget.getGridPosition()
            qp.drawRect(self.x, self.y, self.widgetLength, self.widgetHeight)
            self.update()
            qp.end()

    def mouseReleaseEvent(self, event):
        QApplication.restoreOverrideCursor()


class MyTreeWidget(QTreeWidget):

    def __init__(self, parent=None):
        super(MyTreeWidget, self).__init__(parent)
        self.funk = 0

    def mousePressEvent(self, event):
        super(MyTreeWidget, self).mousePressEvent(event)
#         QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
        self.newWidgetCreatedFlag = False
        self.offset = event.pos()
        self.itemText = gui.deviceList.currentItem().text(0)

        self.funk = getattr(
            gui,
            "create" +
            str(self.itemText),
            None)
        if self.funk and gui.deviceList.currentItem().isSelected():
            QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        #         super(MyTreeWidget, self).mouseMoveEvent(event)
        if (event.buttons() & Qt.LeftButton) and gui.deviceList.currentItem().isSelected():
            self.distance = (event.pos() - self.offset).manhattanLength()
            if self.distance > QApplication.startDragDistance():

                if self.funk:
                    if not self.newWidgetCreatedFlag:
                        self.newWidgetCreatedFlag = True
                        self.funk()

                    gui.widget.moveWidget(
                        QGroupBox.mapToParent(
                            self,
                            event.pos() -
                            self.offset))

    def mouseReleaseEvent(self, event):
        super(MyTreeWidget, self).mouseReleaseEvent(event)
        QApplication.restoreOverrideCursor()
        if self.funk and self.newWidgetCreatedFlag:

            gui.widget.moveFollowGrid()

        gui.deviceList.clearSelection()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    pal = QPalette
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor.fromRgb(230, 227, 224))
    app.setPalette(pal)

    gui = ConcertGUI()

    gui.show()

    sys.exit(app.exec_())
