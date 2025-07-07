import pint
from pint import Quantity

q = pint.UnitRegistry()

q.define('pixel = 1 * count = px')

q.define('fraction = [] = frac')
q.define('percent = 1e-2 frac = pct')


def has_unit(value: Quantity, unit: str) -> bool:
    """Validates if the provided value has the specified uint"""
    return value.u.__str__() == unit
