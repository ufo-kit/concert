from concert.helpers import Command


class GuiCommand(Command):

    """Start concert GUI."""

    def __init__(self):
        opts = {'session': {'type': str}}
        super(GuiCommand, self).__init__('gui', opts)

    def run(self, session):
        import sys
        from PyQt4.QtGui import QApplication, QPalette, QColor,QStyleFactory
        from concert.gui.main import ConcertGUI
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create("plastique"))
        pal = QPalette
        pal = app.palette()
        pal.setColor(QPalette.Window, QColor.fromRgb(232, 229, 226))
        app.setPalette(pal)
        gui = ConcertGUI(session)
        gui.show()
        sys.exit(app.exec_())
