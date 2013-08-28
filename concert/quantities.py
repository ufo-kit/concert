import pint

q = pint.UnitRegistry()


def numerator_units(quantity):
    """Return the numerator units of a quantity"""
    units = (u for u in quantity.units if quantity.units[u] > 0.0)
    return q.parse_expression(' * '.join(units))


def denominator_units(quantity):
    """Return the denominator units of a quantity"""
    units = (u for u in quantity.units if quantity.units[u] < 0.0)
    return q.parse_expression(' * '.join(units))
