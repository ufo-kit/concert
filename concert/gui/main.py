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
        self.device_tree = DeviceTreeWidget()
        self._grid_lines = []
        self._width = 160
        self.device_tree.setFixedWidth(self._width)
        self.device_tree.header().setStretchLastSection(False)
        self.device_tree.setHeaderItem(QTreeWidgetItem(["Devices"]))
        self.device_tree.setColumnWidth(0, 150)
        self._items_list = {}
        """ Adding items to device tree"""
        for obj in globals():
            if isinstance(globals()[obj], Device):
                name_of_class = globals()[obj].__class__.__name__
                try:
                    str(name_of_class).split("Motor")[1]
                except:
                    if name_of_class not in self._items_list:
                        header = QTreeWidgetItem(
                            self.device_tree, [name_of_class])
                        QTreeWidgetItem(header, [obj])
                        self._items_list[name_of_class] = header
                    else:
                        QTreeWidgetItem(
                            self._items_list[name_of_class],
                            [obj])
                else:
                    if "Motors" not in self._items_list:
                        _header_motor = QTreeWidgetItem(
                            self.device_tree,
                            ["Motors"])
                        self._items_list["Motors"] = _header_motor
                        self.device_tree.setItemExpanded(_header_motor, True)
                    name_of_class = str(name_of_class).split("Motor")[0]
                    if name_of_class not in self._items_list:
                        header = QTreeWidgetItem(
                            _header_motor,
                            [name_of_class])
                        QTreeWidgetItem(header, [obj])
                        self._items_list[name_of_class] = header
                    else:
                        QTreeWidgetItem(
                            self._items_list[name_of_class],
                            [obj])
                self.device_tree.setItemExpanded(header, True)
        exit_action = QAction(
            QIcon.fromTheme("application-exit"),
            '&Exit',
            self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        hide_tree_action = QAction(
            QIcon.fromTheme("zoom-fit-best"),
            '&Hide/Show list of widgets',
            self)
        hide_tree_action.setShortcut('Ctrl+H')
        hide_tree_action.setStatusTip('Hide/Show list of widgets')
        hide_tree_action.triggered.connect(self._hide_widgets_list)
        self._menubar = QMenuBar()
        file_menu = self._menubar.addMenu('&File')
        file_menu.addAction(exit_action)
        view_menu = self._menubar.addMenu('&View')
        view_menu.addAction(hide_tree_action)
        self._main_layout = QVBoxLayout()
        self._main_layout.addWidget(self._menubar, 0)
        self._main_layout.addWidget(self.device_tree, 1, Qt.AlignLeft)
        self.setLayout(self._main_layout)
        self.setWindowTitle("Concert GUI")
        self.resize(1024, 500)
        self.widget = WidgetPattern("")

    def createLinear(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createContinuousLinear(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createRotation(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createContinuousRotation(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createLightSource(self, nameOfWidget):
        self.widget = LightSourceWidget(nameOfWidget,
                                        globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createPositioner(self, nameOfWidget):
        self.widget = PositionerWidget(nameOfWidget,
                                       globals()[str(nameOfWidget)], self)
        self.widget().show()

    def createShutter(self, nameOfWidget):
        self.widget = ShutterWidget(nameOfWidget,
                                    globals()[str(nameOfWidget)], self)
        self.widget().show()

    def paintEvent(self, event):
        if self.widget.get_shadow_status():
            qp = QPainter()
            qp.begin(self)
            qp.setBrush(QColor("#ffcccc"))
            qp.setPen(Qt.NoPen)
            self.x, self.y = self.widget.get_grid_position()
            qp.drawRect(
                self.x,
                self.y,
                self.widget.widgetLength,
                self.widget.height())
            qp.setPen(QPen(Qt.black, 1, Qt.DashDotLine))
            qp.drawLines(self._grid_lines)
            self.update()
            qp.end()

    def resizeEvent(self, event):
        for i in xrange(int(self.width() / self.widget.grid_x_step)):
            x = i * self.widget.grid_x_step + 10
            if x >= 130:
                self._grid_lines.append(QLine(x, 0, x, self.height()))
        for i in xrange(int(self.height() / self.widget.grid_y_step)):
            y = i * self.widget.grid_y_step
            self._grid_lines.append(QLine(150, y, self.width(), y))

    def mouseReleaseEvent(self, event):
        QApplication.restoreOverrideCursor()

    def _hide_widgets_list(self):
        self._width = 160 - self._width
        self.device_tree.setFixedWidth(self._width)

    def _close_button_clicked(self):
        sender = self.sender().parent()
        item = self.device_tree.findItems(
            sender.name.text(),
            Qt.MatchExactly | Qt.MatchRecursive)
        item[-1].setDisabled(False)
        sender.close()


class DeviceTreeWidget(QTreeWidget):

    """Determines device tree widget behavior"""

    def __init__(self, parent=None):
        super(DeviceTreeWidget, self).__init__(parent)
        self._func = 0

    def mousePressEvent(self, event):
        super(DeviceTreeWidget, self).mousePressEvent(event)
        if (event.buttons() & Qt.LeftButton) and not gui.device_tree.currentItem().isDisabled():
            self._new_widget_created_flag = False
            self._offset = event.pos()
            try:
                self.itemText = gui.device_tree.currentItem().parent().text(0)
            except:
                self.itemText = None
            self._func = getattr(
                gui,
                "create" +
                str(self.itemText), None)
            if self._func:
                QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and not gui.device_tree.currentItem().isDisabled():
            _distance = (event.pos() - self._offset).manhattanLength()
            if _distance > QApplication.startDragDistance():
                if self._func:
                    if not self._new_widget_created_flag:
                        self._new_widget_created_flag = True
                        self._func(gui.device_tree.currentItem().text(0))
                        gui.widget.close_button.clicked.connect(
                            gui._close_button_clicked)
                    gui.widget.move_widget(
                        QTreeWidget.mapToParent(
                            self,
                            event.pos() - QPoint(140, 0)))

    def mouseReleaseEvent(self, event):
        super(DeviceTreeWidget, self).mouseReleaseEvent(event)
        QApplication.restoreOverrideCursor()
        if self._func and self._new_widget_created_flag:
            gui.widget.move_by_grid()
            gui.device_tree.currentItem().setDisabled(True)
            self._new_widget_created_flag = False

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
