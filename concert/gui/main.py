from widgets import *
from spyderlib.widgets.internalshell import InternalShell
from concert.session.management import load


class ConcertGUI(QtGui.QMainWindow):

    def __init__(self, session_name, parent=None):
        super(ConcertGUI, self).__init__(parent)
        self.session = load(session_name)
        self._cursor = QtGui.QCursor
        self._start_line_point = QtCore.QPoint()
        self.dic_index = 0
        self.device_tree = TreeWidget(self)
        self._width = 200
        self._grid_lines = []
        self.lines_info = {}
        self._add_device_tree()
        exit_action = QtGui.QAction(
            QtGui.QIcon.fromTheme("application-exit"),
            '&Exit',
            self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        hide_tree_action = QtGui.QAction(
            QtGui.QIcon.fromTheme("zoom-fit-best"),
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
        self._add_function_tree()
        view_menu.addAction(hide_tree_action)
        self.setWindowTitle("Concert GUI")
        self.resize(1300, 1000)
        self.widget = WidgetPattern("")
        self.initial_port = None

    def set_current_widget(self):
        self.widget = self.sender()

    def _add_function_tree(self):
        self.function_tree = TreeWidget(self)
        self.function_tree.setHeaderItem(QtGui.QTreeWidgetItem(["function"]))
        for name in dir(self.session):
            object = getattr(self.session, name)
            try:
                if object._isfunction:
                    QtGui.QTreeWidgetItem(self.function_tree, [str(name)])
            except:
                pass
        self.function_tree.adjustSize()
        self.function_tree.resizeColumnToContents(0)
        dock = QtGui.QDockWidget("Functions", self)
        dock.setWidget(self.function_tree)
        dock.adjustSize()
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _add_device_tree(self):
        self.device_tree.setMaximumWidth(self._width)
        self.device_tree.setHeaderItem(QtGui.QTreeWidgetItem(["Devices"]))
        self.device_tree.header().setStretchLastSection(False)
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
                        header = QtGui.QTreeWidgetItem(
                            self.device_tree, [name_of_class])
                        QtGui.QTreeWidgetItem(header, [name])
                        self._items_list[name_of_class] = header
                    else:
                        QtGui.QTreeWidgetItem(
                            self._items_list[name_of_class],
                            [name])
                else:
                    if "Motors" not in self._items_list:
                        _header_motor = QtGui.QTreeWidgetItem(
                            self.device_tree,
                            ["Motors"])
                        self._items_list["Motors"] = _header_motor
                        self.device_tree.setItemExpanded(_header_motor, True)
                    name_of_class = str(name_of_class).split("Motor")[0]
                    if name_of_class not in self._items_list:
                        header = QtGui.QTreeWidgetItem(
                            _header_motor,
                            [name_of_class])
                        QtGui.QTreeWidgetItem(header, [name])
                        self._items_list[name_of_class] = header
                    else:
                        QtGui.QTreeWidgetItem(
                            self._items_list[name_of_class],
                            [name])
                self.device_tree.setItemExpanded(header, True)
        self.device_tree.resizeColumnToContents(0)

    def _create_terminal(self):
        font = QtGui.QFont()
        font.setStyleHint(QtGui.QFont.Monospace)
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
                globals().update({name:object})
            try:
                if object._isfunction:
                    globals().update({name:object})
            except:
                pass
            \n\n""")
        cons.execute_command("cls")
        console_dock = QtGui.QDockWidget("Console", self)
        console_dock.setWidget(cons)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, console_dock)

    def create_function(self, nameOfWidget):
        self.widget = FunctionWidget(str(nameOfWidget), self)
        self.widget.show()

    def create_Motors(self, nameOfWidget):
        self.widget = MotorWidget(
            str(nameOfWidget),
            getattr(self.session, str(nameOfWidget)), self)
        self.widget.show()

    def create_LightSource(self, nameOfWidget):
        self.widget = LightSourceWidget(
            nameOfWidget,
            getattr(
                self.session,
                str(nameOfWidget)),
            self)
        self.widget.show()

    def create_Positioner(self, nameOfWidget):
        self.widget = PositionerWidget(
            nameOfWidget,
            getattr(
                self.session,
                str(nameOfWidget)),
            self)
        self.widget.show()

    def create_Shutter(self, nameOfWidget):
        self.widget = ShutterWidget(
            nameOfWidget,
            getattr(
                self.session,
                str(nameOfWidget)),
            self)
        self.widget.show()

    def create_Camera(self, nameOfWidget):
        self.widget = CameraWidget(
            nameOfWidget,
            getattr(
                self.session,
                str(nameOfWidget)),
            self)
        self.widget.show()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.widget.get_shadow_status():
            qp.setBrush(QtGui.QColor("#ffcccc"))
            qp.setPen(QtCore.Qt.NoPen)
            x, y = self.widget.get_grid_position()
            qp.drawRect(x, y, self.widget.width(), self.widget.height())
            qp.setPen(QtGui.QPen(QtCore.Qt.black, 1, QtCore.Qt.DashDotLine))
            qp.drawLines(self._grid_lines)
        qp.setPen(QtGui.QPen(QtCore.Qt.black, 1))
        if self.widget.get_draw_line_status():
            qp.drawLine(
                self._start_line_point, self.mapFromGlobal(self._cursor.pos()))
        for line in self.lines_info.itervalues():
            qp.drawLine(line)
        self.update()
        qp.end()

    def resizeEvent(self, event):
        for i in xrange(int(self.width() / self.widget.grid_x_step)):
            x = i * self.widget.grid_x_step
            if x >= 130:

                self._grid_lines.append(QtCore.QLine(x, 0, x, self.height()))
        for i in xrange(int(self.height() / self.widget.grid_y_step)):
            y = i * self.widget.grid_y_step
            self._grid_lines.append(QtCore.QLine(150, y, self.width(), y))

    def mouseReleaseEvent(self, event):
        QtGui.QApplication.restoreOverrideCursor()
        widget = QtGui.QApplication.widgetAt(self._cursor.pos())
        if (str(widget.__class__.__name__) == "PortWidget") and (
                self.initial_port is not None):
            self.connection_complited(
                QtGui.QApplication.widgetAt(self._cursor.pos()))
            self.initial_port = None

    def _hide_widgets_list(self):
        self._width = 200 - self._width
        self.device_tree.setFixedWidth(self._width)

    def _close_button_clicked(self):
        sender = self.sender().parent()
        item = self.device_tree.findItems(
            sender.name.text(),
            QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
        if item == []:
            item = self.function_tree.findItems(
                sender.name.text(),
                QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
        item[-1].setDisabled(False)
        sender.close()

    def new_connection(self, start_point):
        self._start_line_point = start_point
        self.initial_port = self.sender()

    def connection_complited(self, port):
        '''Create a connection between widgets'''
        if port.dic_index is None and self.initial_port.parent(
        ) is not port.parent():
            new_line = Line(self._start_line_point, self.mapFromGlobal(
                port.mapToGlobal(port.connection_point)))
            finish_point = self.mapFromGlobal(
                port.mapToGlobal(
                    port.connection_point))
            self.lines_info[self.dic_index] = new_line
            new_line.start_point = self._start_line_point
            new_line.finish_point = finish_point
            new_line.start_port = self.initial_port
            new_line.finish_port = port
            self.initial_port.is_start_point = True
            port.is_start_point = False
            self.initial_port.dic_index = port.dic_index = self.dic_index
            self.dic_index += 1


class TreeWidget(QtGui.QTreeWidget):

    """Determines device tree widget behavior"""

    def __init__(self, parent=None):
        super(TreeWidget, self).__init__(parent)
        self._func = 0
        self.gui = self.parent()
        self.setHeaderHidden(True)

    def mousePressEvent(self, event):
        super(TreeWidget, self).mousePressEvent(event)
        if self.currentItem() is not None:
            if (event.buttons() & QtCore.Qt.LeftButton) and not (
                    self.currentItem().isDisabled()):
                self._new_widget_created_flag = False
                self._offset = event.pos()
                if self.headerItem().text(0) == "function":
                    self.itemText = "function"
                else:
                    try:
                        self.itemText = self.currentItem().parent().text(
                            0)
                        try:
                            self.itemText = self.currentItem(
                            ).parent().parent().text(0)
                        except:
                            pass
                    except:
                        self.itemText = self.currentItem().text(0)
                self._func = getattr(
                    self.gui,
                    "create_" +
                    str(self.itemText), None)
                if self._func:
                    QtGui.QApplication.setOverrideCursor(
                        QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            self.itemText = str(self.itemText)

    def mouseMoveEvent(self, event):
        if self.currentItem() is not None:
            if (event.buttons() & QtCore.Qt.LeftButton) and not (
                    self.currentItem().isDisabled()):
                _distance = (event.pos() - self._offset).manhattanLength()
                if _distance > QtGui.QApplication.startDragDistance():
                    if self._func and self.currentItem().child(0) is None:
                        if not self._new_widget_created_flag:
                            self._new_widget_created_flag = True
                            self._func(self.currentItem().text(0))
                            self.gui.widget.close_button.clicked.connect(
                                self.gui._close_button_clicked)
                        pos = QtCore.QPoint(int(self.gui.widget.width() / 2), 0)
                        self.gui.widget.move_widget(self.gui.mapFromGlobal(
                            self.gui._cursor.pos()) - pos)

    def mouseReleaseEvent(self, event):
        super(TreeWidget, self).mouseReleaseEvent(event)
        QtGui.QApplication.restoreOverrideCursor()
        if self._func and self._new_widget_created_flag:
            self.gui.widget.move_by_grid()
            self.currentItem().setDisabled(True)
            self._new_widget_created_flag = False

    def sizeHint(self):
        return QtCore.QSize(120, 75)


class Line(QtCore.QLine):

    def __init__(self, start_point=QtCore.QPoint(),
                 finish_point=QtCore.QPoint()):
        super(Line, self).__init__(start_point, finish_point)
        self.start_point = QtCore.QPoint()
        self.start_port = None
        self.finish_point = QtCore.QPoint()
        self.finish_port = None

    def __del__(self):
        self.start_port.dic_index = None
        self.finish_port.dic_index = None


def main():
    import sys
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create("plastique"))
    pal = QtGui.QPalette
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor.fromRgb(232, 229, 226))
    app.setPalette(pal)
    gui = ConcertGUI("new-session")
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
