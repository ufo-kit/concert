from widgets import *
from spyderlib.widgets.internalshell import InternalShell
import concert.session.management as cs
import sip


class ConcertGUI(QtGui.QMainWindow):

    def __init__(self, session_name=None, parent=None):
        super(ConcertGUI, self).__init__(parent)

        self._cursor = QtGui.QCursor
        self._start_line_point = QtCore.QPoint()
        self.line_index = 0
        self._width = 200
        self._grid_lines = []
        self.lines_info = {}
        self.widget = None
        self.initial_port = None
        self.current_layout = "Untitled"
        self.session = session_name
        self.device_tree = TreeWidget(self)
        self.device_tree.setMaximumWidth(self._width)
        self.device_tree.setHeaderItem(QtGui.QTreeWidgetItem(["Devices"]))
        self.device_tree.header().setStretchLastSection(False)
        self.function_tree = TreeWidget(self)
        self.function_tree.setHeaderItem(QtGui.QTreeWidgetItem(["function"]))

        exit_action = QtGui.QAction(
            QtGui.QIcon.fromTheme("application-exit"),
            '&Exit',
            self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        hide_tree_action = QtGui.QAction(
            QtGui.QIcon.fromTheme("zoom-fit-best"),
            '&Device list',
            self)
        hide_tree_action.setShortcut('Ctrl+H')
        hide_tree_action.setStatusTip('widgets list')
        hide_tree_action.triggered.connect(self._hide_widgets_list)
        hide_tree_action.setCheckable(True)
        hide_tree_action.setChecked(True)
        visualization = QtGui.QAction(
            '&Visualization widget', self)
        visualization.triggered.connect(self._create_visualization_widget)
        visualization.setShortcut('Ctrl+R')

        plotting = QtGui.QAction('&Plot widget', self)
        plotting.triggered.connect(self._create_plot_widget)
        plotting.setShortcut('Ctrl+P')

        save_layout = QtGui.QAction('&save', self)
        save_layout.triggered.connect(self.save_layout)
        save_layout.setShortcut('Ctrl+S')

        save_layout_as = QtGui.QAction('&save as', self)
        save_layout_as.triggered.connect(self.save_layout_as)
        save_layout_as.setShortcut('Ctrl+Shift+S')

        open_layout = QtGui.QAction('&open', self)
        open_layout.triggered.connect(self.open_layout)
        open_layout.setShortcut('Ctrl+O')

        open_last_layout = QtGui.QAction('&open previous', self)
        open_last_layout.triggered.connect(self.open_last_layout)
        open_last_layout.setShortcut('Ctrl+Shift+L')

        load_session = QtGui.QAction('&load session', self)
        load_session.triggered.connect(self.load_session)
        load_session.setShortcut('Ctrl+L')

        self._menubar = self.menuBar()
        file_menu = self._menubar.addMenu('&File')
        file_menu.addAction(load_session)

        file_menu.addAction(exit_action)
        view_menu = self._menubar.addMenu('&View')
        view_menu.addAction(hide_tree_action)
        view_menu.addAction(visualization)
        view_menu.addAction(plotting)

        layout_menu = self._menubar.addMenu('&Layout')
        layout_menu.addAction(open_layout)
        layout_menu.addAction(open_last_layout)
        layout_menu.addAction(save_layout)
        layout_menu.addAction(save_layout_as)

        self.console_dock = QtGui.QDockWidget("Console", self)
        self.console_dock.setObjectName("console")
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.console_dock)

        self.initUI(session_name)

        dock = QtGui.QDockWidget("Functions", self)
        dock.setObjectName("functions")
        dock.setWidget(self.function_tree)
        dock.adjustSize()
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self.setCentralWidget(self.device_tree)
        self.set_window_title()
        self.resize(1300, 1000)
        self.showMaximized()
        self.setMinimumSize(self.size())
        if session_name is None:
            load_session.triggered.emit(True)

    def set_window_title(self):
        self.setWindowTitle("%s - Concert GUI" % (self.current_layout))

    def initUI(self, session):
        if not session is None:
            self.session = cs.load(str(session))
            self.device_tree.clear()
            self.function_tree.clear()
            self._add_device_tree()
            self._add_function_tree()
        self._create_terminal()

    def set_current_widget(self):
        self.widget = self.sender()

    def _create_plot_widget(self):
        self.p_widget = PlotWidget("Line plot widget", self)
        self.p_widget.resize(550, 500)
        self.p_widget.move(100, 100)
        self.p_widget.show()

    def _create_visualization_widget(self):
        self.v_widget = VisualizationWidget("3D Visualization", None)
        self.v_widget.show()

    def _add_function_tree(self):
        for name in dir(self.session):
            object = getattr(self.session, name)
            if hasattr(object, "_isfunction"):
                QtGui.QTreeWidgetItem(self.function_tree, [str(name)])

    def _add_device_tree(self):
        self._items_list = {}
        """ Adding items to device tree"""
        for name in dir(self.session):
            object = getattr(self.session, name)
            if isinstance(object, Device):
                name_of_class = object.__class__.__name__
                if "Motor" in name_of_class:
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
                else:
                    if name_of_class not in self._items_list:
                        header = QtGui.QTreeWidgetItem(
                            self.device_tree, [name_of_class])
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
            if hasattr(object,"_isfunction"):
                    globals().update({name:object})
            \n\n""")
        cons.execute_command("cls")
        self.console_dock.setWidget(self.console)

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

        color = QtGui.QColor(64, 64, 64)

        if self.widget is not None and self.widget.get_shadow_status():
            qp.setBrush(QtGui.QColor("#ffcccc"))
            qp.setPen(QtCore.Qt.NoPen)
            x, y = self.widget.get_grid_position()
            qp.drawRect(x, y, self.widget.width(), self.widget.height())
            qp.setPen(QtGui.QPen(color, 1, QtCore.Qt.DotLine))
            qp.drawLines(self._grid_lines)

        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setPen(QtGui.QPen(color, 2))

        if self.widget is not None and self.widget.get_draw_line_status():
            qp.drawLine(self._start_line_point, self.mapFromGlobal(self._cursor.pos()))

        for line in self.lines_info.itervalues():
            qp.drawLine(line)

        self.update()
        qp.end()

    def resizeEvent(self, event):
        for i in xrange(int(self.width() / WidgetPattern.grid_x_step)):
            x = i * WidgetPattern.grid_x_step
            if x >= 0:
                self._grid_lines.append(QtCore.QLine(x, 0, x, self.height()))
        for i in xrange(int(self.height() / WidgetPattern.grid_y_step)):
            y = i * WidgetPattern.grid_y_step
            self._grid_lines.append(QtCore.QLine(0, y, self.width(), y))

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
        if not item == []:
            item[-1].setDisabled(False)
        sender.close()
        self.save_layout("autosave")

    def new_connection(self, start_point):
        self._start_line_point = start_point
        self.initial_port = self.sender()

    def connection_complited(self, port):
        '''Create a connection between widgets'''
        if self._is_connection_allowed(port):
            new_line = Line(self._start_line_point, self.mapFromGlobal(
                port.mapToGlobal(port.connection_point)))
            finish_point = self.mapFromGlobal(
                port.mapToGlobal(
                    port.connection_point))
            self.lines_info[self.line_index] = new_line
            new_line.start_point = self._start_line_point
            new_line.finish_point = finish_point
            new_line.start_port = self.initial_port
            new_line.finish_port = port

            self.initial_port.is_start_point = True
            port.is_start_point = False
            new_line.start_port.port_connected.emit()
            new_line.finish_port.port_connected.emit()
            self.line_index += 1

    def _is_connection_allowed(self, port):
        if hasattr(port.parent(), "object") and isinstance(port.parent().object, Device):
            cond1 = True
        elif len(port.get_line_number()) == 0:
            cond1 = True
        else:
            cond1 = False
        port_parent = self.initial_port.parent()
        if hasattr(port_parent, "object") and isinstance(port_parent.object, Device):
            cond2 = True
        elif len(self.initial_port.get_line_number()) == 0:
            cond2 = True
        else:
            cond2 = False
        cond3 = self.initial_port.parent() is not port.parent()
        return cond1 & cond2 & cond3

    def save_layout(self, file_name=None):
        if not file_name:
            file_name = self.current_layout
        settings = QtCore.QSettings('concert', file_name)
        settings.clear()
        for obj in WidgetPattern.get_instances():
            if obj.isVisible():
                name = obj.name.text()
                settings.beginGroup(name)
                settings.setValue('position', obj.pos())
                settings.endGroup()
        settings.setValue('session', self.session.__name__)
        settings.setValue('state', self.saveState())

    def save_layout_as(self):

        text, ok = QtGui.QInputDialog.getText(self, 'Save layout as',
                                              'Enter layout name:', False, self.current_layout)
        if ok:
            self.current_layout = text
            self.set_window_title()
            self.save_layout(file_name=text)

    def open_layout(self, file=None):
        if not file:
            path = QtCore.QDir().homePath() + "/.config/concert"
            file = QtGui.QFileDialog.getOpenFileName(
                self, "Load layout from file", path, "*.conf")
            if not file == "":
                self.current_layout = file.section('/', -1).split('.')[0]
            else:
                return
        else:
            self.current_layout = file
        self._remove_all_widgets()
        self.set_window_title()
        settings = QtCore.QSettings(
            'concert', self.current_layout)
        session = settings.value('session', type=str)
        if not hasattr(self.session, "__name__") or not self.session.__name__ == session:
            self.initUI(session)
        devices = settings.childGroups()
        for widget_name in devices:
            if widget_name == "Line plot widget":
                settings.beginGroup(widget_name)
                position = settings.value('position', type=QtCore.QPoint)
                settings.endGroup()
                self._create_plot_widget()
                self.p_widget.move(position)
                continue
            item = self.device_tree.findItems(
                widget_name, QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
            if item == []:
                item = self.function_tree.findItems(
                    widget_name, QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
            if item == []:
                continue
            item = item[0]
            tree = item.treeWidget()
            item.setDisabled(True)
            if tree.headerItem().text(0) == "function":
                header_text = "function"
            else:
                header = item.parent()
                if header is not None:
                    if hasattr(header.parent(), "text"):
                        header_text = header.parent().text(0)
                    else:
                        header_text = header.text(0)
            self._func = getattr(
                self,
                "create_" +
                str(header_text), None)
            self._func(widget_name)
            settings.beginGroup(widget_name)
            position = settings.value('position', type=QtCore.QPoint)
            settings.endGroup()
            self.widget.move(position)

    def load_session(self):
        session, ok = QtGui.QInputDialog.getItem(self, 'Load session',
                                                 'Choose session:', cs.get_existing())
        if ok:
            self._remove_all_widgets()
            self.initUI(session)

    def _remove_all_widgets(self):
        for widget in WidgetPattern.get_instances():
            if not sip.isdeleted(widget):
                widget.close_button.clicked.emit(True)

    def open_last_layout(self):
        self.open_layout("autosave")


class TreeWidget(QtGui.QTreeWidget):

    """Determines device tree widget behavior"""

    def __init__(self, parent=None):
        super(TreeWidget, self).__init__(parent)
        self._func = 0
        self.gui = self.parent()
        self.setHeaderHidden(True)
        self.header_text = None

    def mousePressEvent(self, event):
        super(TreeWidget, self).mousePressEvent(event)
        if self.currentItem() is not None:
            if (event.buttons() & QtCore.Qt.LeftButton) and not (
                    self.currentItem().isDisabled()):
                self._new_widget_created_flag = False
                self._offset = event.pos()

                if self.headerItem().text(0) == "function":
                    self.header_text = "function"
                else:
                    header = self.currentItem().parent()
                    if header is not None:
                        if hasattr(header.parent(), "text"):
                            self.header_text = header.parent().text(0)
                        else:
                            self.header_text = header.text(0)
                self._func = getattr(
                    self.gui,
                    "create_" +
                    str(self.header_text), None)
                if self._func:
                    QtGui.QApplication.setOverrideCursor(
                        QtGui.QCursor(QtCore.Qt.ClosedHandCursor))

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
                        pos = QtCore.QPoint(
                            int(self.gui.widget.width() / 2), 0)
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


def main():
    import sys
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create("plastique"))
    pal = QtGui.QPalette
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor.fromRgb(232, 229, 226))
    app.setPalette(pal)
    ConcertGUI("new-session")
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
