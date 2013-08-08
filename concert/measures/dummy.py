""" This module provides different dummy measures."""


class DummyGradientMeasure(object):

    """Gradient measure that returns a quadratic fall-off of *parameter* from
    *max_position*."""

    def __init__(self, parameter, max_position):
        self.max_position = max_position
        self._max_gradient = 1e4
        self._param = parameter

    def __call__(self):
        value = self._param.get().result()
        position = value.to(self.max_position.units).magnitude
        return self._max_gradient - \
            (position - self.max_position.magnitude) ** 2
