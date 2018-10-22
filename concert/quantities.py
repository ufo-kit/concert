import pint

q = pint.UnitRegistry()

q.define('pixel = 1 * count = px')

q.define('fraction = [] = frac')
q.define('percent = 1e-2 frac = pct')
