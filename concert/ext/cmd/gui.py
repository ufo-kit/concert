from concert.helpers import Command


class GuiCommand(Command):

    """Start concert GUI."""

    def __init__(self):

        opts = {'session': {'type': str}}
        super(GuiCommand, self).__init__('gui', opts)

    def run(self, session):
        import sys
        from PyQt4.QtGui import QApplication, QPalette, QColor
        from concert.gui.main import ConcertGUI
        app = QApplication(sys.argv)
        pal = QPalette
        pal = app.palette()
        pal.setColor(QPalette.Window, QColor.fromRgb(230, 227, 224))
        app.setPalette(pal)
        gui = ConcertGUI(session)
        gui.show()
        sys.exit(app.exec_())
