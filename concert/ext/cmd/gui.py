
import sys
import os
import concert

from concert.helpers import Command, Bunch


class GuiCommand(Command):

    """Start concert GUI."""

    def __init__(self):

        opts = {'session': {'type': str}}
        super(GuiCommand, self).__init__('gui', opts)

    def run(self, session):
        path = os.path.dirname(concert.__file__)
        os.system("python " + path + "/gui/main.py 1")
