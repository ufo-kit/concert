plugins = []

try:
    from . import spyder

    plugins.append(spyder.SpyderCommand())
except ImportError:
    pass

try:
    from . import gui

    plugins.append(gui.GuiCommand())
except ImportError:
    pass
