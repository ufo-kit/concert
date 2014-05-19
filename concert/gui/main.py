'''
Created on Apr 28, 2014

@author: Pavel Rybalko (ANKA)
'''

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from widgets import *


class ConcertGUI(QWidget):

    def __init__(self, parent=None):
        super(ConcertGUI, self).__init__(parent)
        self.cursor = QCursor
        self.numberOfMotorWidget = 1
        self.deviceTree = DeviceTreeWidget()
        self.deviceTree.setFixedWidth(150)
        self.deviceTree.header().setStretchLastSection(False)
        self.deviceTree.setHeaderItem(QTreeWidgetItem(["Devices"]))
        self.deviceTree.setColumnWidth(0, 140)
        self.mainLayout = QHBoxLayout()
        self.fieldLayout = QGridLayout()
        self.mainLayout.addWidget(self.deviceTree, 0, Qt.AlignLeft)
        self.mainLayout.addLayout(self.fieldLayout)
        self.setLayout(self.mainLayout)
        self.setWindowTitle("Concert GUI")
        self.resize(1024, 500)
        self.widget = WidgetPattern("")
        self.itemsList = {}
        """ Adding items to device tree"""
        for obj in globals():
            if isinstance(globals()[obj], Device):
                nameOfClass = globals()[obj].__class__.__name__
                if nameOfClass not in self.itemsList:
                    header = QTreeWidgetItem(
                        self.deviceTree, [
                            globals()[obj].__class__.__name__])
                    QTreeWidgetItem(header, [obj])
                    self.itemsList[globals()[obj].__class__.__name__] = header
                else:
                    QTreeWidgetItem(self.itemsList[nameOfClass], [obj])
                self.deviceTree.setItemExpanded(header,True)

    def createLinearMotor(self, nameOfWidget):

        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()
#         self.numberOfMotorWidget += 1

    def createContinuousLinearMotor(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createRotationMotor(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createContinuousRotationMotor(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createCameras(self, nameOfWidget):
        self.button = QPushButton("Camera")
        self.button.setStyleSheet("background-color: green")
        self.label = QLabel("Camera")
        self.line = QLineEdit()
        self.widget = WidgetPattern("Camera", self)
        self.motorvbox = QVBoxLayout()
        self.motorvbox.addWidget(self.label)
        self.motorvbox.addWidget(self.line)
        self.motorvbox.addWidget(self.button)
        self.widget.setLayout(self.motorvbox)
        self.widget.setFixedSize(200, 100)
        self.setObjectName('CameraWidget%d')
        self.widget.show()

    def createLightSource(self, nameOfWidget):
        global count
        self.widget = LightSourceWidget(nameOfWidget, self)
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


class DeviceTreeWidget(QTreeWidget):

    def __init__(self, parent=None):
        super(DeviceTreeWidget, self).__init__(parent)
        self.funk = 0
        self.cursor = QCursor
        

    def mousePressEvent(self, event):
        super(DeviceTreeWidget, self).mousePressEvent(event)
        if (event.buttons() & Qt.LeftButton) and not gui.deviceTree.currentItem().isDisabled():
            self.newWidgetCreatedFlag = False
            self.offset = event.pos()
            try:
                self.itemText = gui.deviceTree.currentItem().parent().text(0)
            except:
                self.itemText = None
            self.funk = getattr(
                gui,
                "create" +
                str(self.itemText), None)
            if self.funk:
                QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and not gui.deviceTree.currentItem().isDisabled():
            self.distance = (event.pos() - self.offset).manhattanLength()
            if self.distance > QApplication.startDragDistance():
                if self.funk:
                    if not self.newWidgetCreatedFlag:
                        self.newWidgetCreatedFlag = True
                        self.funk(gui.deviceTree.currentItem().text(0))
                    gui.widget.moveWidget(
                            QTreeWidget.mapToParent(
                            self,                      
                            event.pos()-QPoint(140,0)))

    def mouseReleaseEvent(self, event):
        super(DeviceTreeWidget, self).mouseReleaseEvent(event)
        QApplication.restoreOverrideCursor()
        if self.funk and self.newWidgetCreatedFlag:
            gui.widget.moveFollowGrid()
            gui.deviceTree.currentItem().setDisabled(True)
            self.newWidgetCreatedFlag = False

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
