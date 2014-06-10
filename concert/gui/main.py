'''
Created on Apr 28, 2014

@author: Pavel Rybalko (ANKA)
'''

from PyQt4.QtGui import QPalette, QColor, QMainWindow, QTreeWidgetItem, QAction
from PyQt4.QtGui import QTreeWidget, QFont, QDockWidget, QPen, QPainter
from PyQt4.QtCore import QLine
from widgets import *
from spyderlib.widgets.internalshell import InternalShell


class ConcertGUI(QMainWindow):

    def __init__(self, session_name, parent=None):
        super(ConcertGUI, self).__init__(parent)
        self.session = load(session_name)
        self.device_tree = DeviceTreeWidget(self)
        self._width = 200
        self._grid_lines = []
        self._add_device_tree()
        exit_action = QAction(
            QIcon.fromTheme("application-exit"),
            '&Exit',
            self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        hide_tree_action = QAction(
            QIcon.fromTheme("zoom-fit-best"),
            '&Widgets list',
            self)
        hide_tree_action.setShortcut('Ctrl+H')
        hide_tree_action.setStatusTip('widgets list')
        hide_tree_action.triggered.connect(self._hide_widgets_list)
        hide_tree_action.setCheckable(True)
        hide_tree_action.setChecked(True)
        self._menubar = self.menuBar()
        file_menu = self._menubar.addMenu('&File')
        file_menu.addAction(exit_action)
        view_menu = self._menubar.addMenu('&View')
        self._create_terminal()
        self.setCentralWidget(self.device_tree)
        view_menu.addAction(hide_tree_action)
        self.setWindowTitle("Concert GUI")
        self.resize(1024, 1000)
        self.widget = WidgetPattern("")

    def _add_device_tree(self):
        self.device_tree.setMaximumWidth(self._width)
        self.device_tree.header().setStretchLastSection(False)
        self.device_tree.setHeaderItem(QTreeWidgetItem(["Devices"]))
        self._items_list = {}
        """ Adding items to device tree"""
        for name in dir(self.session):
            object = getattr(self.session, name)
            if isinstance(object, Device):
                name_of_class = object.__class__.__name__
                try:
                    str(name_of_class).split("Motor")[1]
                except:
                    if name_of_class not in self._items_list:
                        header = QTreeWidgetItem(
                            self.device_tree, [name_of_class])
                        QTreeWidgetItem(header, [name])
                        self._items_list[name_of_class] = header
                    else:
                        QTreeWidgetItem(
                            self._items_list[name_of_class],
                            [name])
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
                        QTreeWidgetItem(header, [name])
                        self._items_list[name_of_class] = header
                    else:
                        QTreeWidgetItem(
                            self._items_list[name_of_class],
                            [name])
                self.device_tree.setItemExpanded(header, True)
        self.device_tree.resizeColumnToContents(0)

    def _create_terminal(self):
        font = QFont()
        font.setStyleHint(QFont.Monospace)
        ns = {'win': self, 'concert': self.session, '': self.session}
        self.console = cons = InternalShell(
            self, namespace=ns, multithreaded=False)
        cons.set_font(font)
        cons.set_codecompletion_auto(True)
        cons.execute_command("from concert.devices.base import Device")
        cons.execute_command("from concert.quantities import q")
        cons.execute_lines("""for name in dir(concert):
            object = getattr(concert, name) 
            if isinstance(object, Device):
                globals().update({name:object}) \n\n""")
        cons.execute_command("cls")
        console_dock = QDockWidget("Console", self)
        console_dock.setWidget(cons)
        self.addDockWidget(Qt.BottomDockWidgetArea, console_dock)

    def createMotors(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            getattr(self.session, str(nameOfWidget)), self)
        self.widget.show()

    def createLightSource(self, nameOfWidget):
        self.widget = LightSourceWidget(nameOfWidget,
                                        getattr(self.session, str(nameOfWidget)), self)
        self.widget.show()

    def createPositioner(self, nameOfWidget):
        self.widget = PositionerWidget(nameOfWidget,
                                       getattr(self.session, str(nameOfWidget)), self)
        self.widget.show()

    def createShutter(self, nameOfWidget):
        self.widget = ShutterWidget(nameOfWidget,
                                    getattr(self.session, str(nameOfWidget)), self)
        self.widget.show()

    def createCamera(self, nameOfWidget):
        self.widget = CameraWidget(nameOfWidget,
                                   getattr(self.session, str(nameOfWidget)), self)
        self.widget.show()

    def paintEvent(self, event):
        if self.widget.get_shadow_status():
            qp = QPainter()
            qp.begin(self)
            qp.setBrush(QColor("#ffcccc"))
            qp.setPen(Qt.NoPen)
            x, y = self.widget.get_grid_position()
            qp.drawRect(x, y, 280, 100)
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
        self._width = 200 - self._width
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
        self.gui = self.parent()

    def mousePressEvent(self, event):
        super(DeviceTreeWidget, self).mousePressEvent(event)
        if (event.buttons() & Qt.LeftButton) and not (
                self.gui.device_tree.currentItem().isDisabled()):
            self._new_widget_created_flag = False
            self._offset = event.pos()
            try:
                self.itemText = self.gui.device_tree.currentItem().parent().text(
                    0)
                try:
                    self.itemText = self.gui.device_tree.currentItem(
                    ).parent().parent().text(0)
                except:
                    pass
            except:
                self.itemText = None
            self._func = getattr(
                self.gui,
                "create" +
                str(self.itemText), None)
            if self._func:
                QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
        self.itemText = str(self.itemText)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and not (
                self.gui.device_tree.currentItem().isDisabled()):
            _distance = (event.pos() - self._offset).manhattanLength()
            if _distance > QApplication.startDragDistance():
                if self._func and self.gui.device_tree.currentItem().child(0) is None:
                    if not self._new_widget_created_flag:
                        self._new_widget_created_flag = True
                        self._func(self.gui.device_tree.currentItem().text(0))
                        self.gui.widget.close_button.clicked.connect(
                            self.gui._close_button_clicked)
                    self.gui.widget.move_widget(
                        event.pos() - QPoint(140, 0))

    def mouseReleaseEvent(self, event):
        super(DeviceTreeWidget, self).mouseReleaseEvent(event)
        QApplication.restoreOverrideCursor()
        if self._func and self._new_widget_created_flag:
            self.gui.widget.move_by_grid()
            self.gui.device_tree.currentItem().setDisabled(True)
            self._new_widget_created_flag = False


def main():
    import sys
    app = QApplication(sys.argv)
    pal = QPalette
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor.fromRgb(230, 227, 224))
    app.setPalette(pal)
    gui = ConcertGUI("new-session")
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
