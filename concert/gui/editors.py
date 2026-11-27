"""Small editor adapters: display, parse, and capability without device I/O."""
from PyQt6.QtWidgets import QLabel, QLineEdit, QCheckBox, QComboBox
from pint.errors import PintError

from concert.base import QuantityValue, SelectionValue
from concert.quantities import q


class ScalarEditor:
    def __init__(self, param):
        self.param = param
        self.data_type = None
        self.widget = QLineEdit() if param.writable else QLabel()

    @property
    def editable(self):
        return self.param.writable and self.data_type in (str, int, float, bool)

    def display(self, value):
        self.data_type = type(value)
        want_checkbox = self.param.writable and type(value) is bool
        if want_checkbox != isinstance(self.widget, QCheckBox):
            self.widget = QCheckBox('Enabled') if want_checkbox else (
                QLineEdit() if self.param.writable else QLabel())
        if isinstance(self.widget, QCheckBox):
            self.widget.setChecked(value)
        else:
            self.widget.setText(str(value))

    def parse(self):
        if not self.editable:
            raise ValueError('Read a supported scalar value before writing, or supply an editor')
        return self.widget.isChecked() if self.data_type is bool else self.data_type(self.widget.text())


class QuantityEditor(ScalarEditor):
    @property
    def editable(self):
        return self.param.writable

    def display(self, value):
        self.widget.setText(str(value.to(self.param.unit).magnitude) if value is not None else 'None')

    def parse(self):
        try:
            value = q.Quantity(self.widget.text())
            if value.dimensionless and not self.param.unit.dimensionless:
                value *= self.param.unit
            return value.to(self.param.unit)
        except (PintError, SyntaxError, ValueError, TypeError) as error:
            raise ValueError(f'Invalid quantity: {error}') from error


class SelectionEditor:
    def __init__(self, param):
        self.param = param
        self.widget = QComboBox()
        # Store Python values separately: QVariant may convert custom types.
        self.values = param.values
        for value in self.values:
            self.widget.addItem(str(value))
        self.widget.setCurrentIndex(-1)

    @property
    def editable(self):
        return self.param.writable

    def display(self, value):
        self.widget.setCurrentIndex(next((i for i, item in enumerate(self.values)
                                          if item == value), -1))

    def parse(self):
        index = self.widget.currentIndex()
        if index < 0:
            raise ValueError('Choose a valid selection before writing')
        return self.values[index]


class EditorRegistry:
    """Map parameter classes to adapter factories, with most-specific precedence."""
    def __init__(self):
        self._factories = [(SelectionValue, SelectionEditor), (QuantityValue, QuantityEditor)]

    def register(self, parameter_type, factory):
        self._factories.insert(0, (parameter_type, factory))

    def create(self, param, override=None):
        if override is not None:
            return override(param)
        for parameter_type, factory in self._factories:
            if isinstance(param, parameter_type):
                return factory(param)
        return ScalarEditor(param)
