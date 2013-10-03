"""Qt Widget."""
try:
    from PyQt4 import QtGui
except ImportError:
    print("PyQt4 is not installed")


def make(device):
    """Create a Qt widget off of a *device*."""
    layout = QtGui.QGridLayout()

    for row, param in enumerate(device):
        name_widget = QtGui.QLabel(str(param.name))
        layout.addWidget(name_widget, row, 0)

        value = param.get().result()
        value_widget = None

        if isinstance(value, str):
            value_widget = QtGui.QLabel()
            value_widget.setText(value)
        else:
            value_widget = QtGui.QDoubleSpinBox()
            value_widget.setValue(value)

        layout.addWidget(value_widget, row, 1)

    return layout
