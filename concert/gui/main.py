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
        self.device_tree = device_treeWidget()
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
                _name_of_class = globals()[obj].__class__.__name__
                try:
                    str(_name_of_class).split("Motor")[1]
                except:
                    if _name_of_class not in self._items_list:
                        _header = QTreeWidgetItem(
                            self.device_tree, [_name_of_class])
                        QTreeWidgetItem(_header, [obj])
                        self._items_list[_name_of_class] = _header
                    else:
                        QTreeWidgetItem(
                            self._items_list[_name_of_class],
                            [obj])
                else:
                    if "Motors" not in self._items_list:
                        _header_motor = QTreeWidgetItem(
                            self.device_tree,
                            ["Motors"])
                        self._items_list["Motors"] = _header_motor
                        self.device_tree.setItemExpanded(_header_motor, True)
                    _name_of_class = str(_name_of_class).split("Motor")[0]
                    if _name_of_class not in self._items_list:
                        _header = QTreeWidgetItem(
                            _header_motor,
                            [_name_of_class])
                        QTreeWidgetItem(_header, [obj])
                        self._items_list[_name_of_class] = _header
                    else:
                        QTreeWidgetItem(
                            self._items_list[_name_of_class],
                            [obj])
                self.device_tree.setItemExpanded(_header, True)
        _exit_action = QAction(
            QIcon.fromTheme("application-exit"),
            '&Exit',
            self)
        _exit_action.setShortcut('Ctrl+Q')
        _exit_action.setStatusTip('Exit application')
        _exit_action.triggered.connect(self.close)
        _hide_tree_action = QAction(
            QIcon.fromTheme("zoom-fit-best"),
            '&Hide/Show list of widgets',
            self)
        _hide_tree_action.setShortcut('Ctrl+H')
        _hide_tree_action.setStatusTip('Hide/Show list of widgets')
        _hide_tree_action.triggered.connect(self._hide_widgets_list)
        self._menubar = QMenuBar()
        _file_menu = self._menubar.addMenu('&File')
        _file_menu.addAction(_exit_action)
        _view_menu = self._menubar.addMenu('&View')
        _view_menu.addAction(_hide_tree_action)
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
             globals()[str(nameOfWidget)],self)
        self.widget().show()
        

    def createPositioner(self, nameOfWidget):
        self.widget = PositionerWidget(nameOfWidget,
             globals()[str(nameOfWidget)],self)
        self.widget().show()

    def paintEvent(self, event):
        if self.widget.get_shadow_status():
            _qp = QPainter()
            _qp.begin(self)
            _qp.setBrush(QColor("#ffcccc"))
            _qp.setPen(Qt.NoPen)
            self.x, self.y = self.widget.get_grid_position()
            _qp.drawRect(
                self.x,
                self.y,
                self.widget.widgetLength,
                self.widget.height())
            _qp.setPen(QPen(Qt.black, 1, Qt.DashDotLine))
            _qp.drawLines(self._grid_lines)
            self.update()
            _qp.end()

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
        _sender=self.sender().parent()
        _item=self.device_tree.findItems(_sender.name.text(), Qt.MatchExactly | Qt.MatchRecursive)
        _item[-1].setDisabled(False)
        _sender.close()


class device_treeWidget(QTreeWidget):

    """Determines device tree widget behavior"""

    def __init__(self, parent=None):
        super(device_treeWidget, self).__init__(parent)
        self._func = 0

    def mousePressEvent(self, event):
        super(device_treeWidget, self).mousePressEvent(event)
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
                        gui.widget.close_button.clicked.connect(gui._close_button_clicked)
                    gui.widget.move_widget(
                        QTreeWidget.mapToParent(
                            self,
                            event.pos() - QPoint(140, 0)))

    def mouseReleaseEvent(self, event):
        super(device_treeWidget, self).mouseReleaseEvent(event)
        QApplication.restoreOverrideCursor()
        if self._func and self._new_widget_created_flag:
            gui.widget.move_by_grid()
            gui.device_tree.currentItem().setDisabled(True)
            self._new_widget_created_flag = False

if __name__ == '__main__':
    import sys
    _app = QApplication(sys.argv)
    _pal = QPalette
    _pal = _app.palette()
    _pal.setColor(QPalette.Window, QColor.fromRgb(230, 227, 224))
    _app.setPalette(_pal)
    gui = ConcertGUI()
    gui.show()
    sys.exit(_app.exec_())
