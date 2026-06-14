from types import SimpleNamespace

import numpy as np

from concert.ext.viewers import _PyQtGraphUpdater


class _Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y


class _ImageItem:
    def __init__(self, image, point):
        self.image = image
        self._point = point

    def sceneBoundingRect(self):
        return SimpleNamespace(contains=lambda event: True)

    def mapFromScene(self, event):
        return self._point


class _View:
    def __init__(self):
        self.title = None
        self.bold = None

    def setTitle(self, title, bold=None):
        self.title = title
        self.bold = bold


def test_pyqtgraph_mouse_moved_rgb_image():
    updater = _PyQtGraphUpdater(None, title='RGB')
    plot_view = _View()
    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    updater.view = SimpleNamespace(
        imageItem=_ImageItem(image, _Point(1, 0)),
        view=plot_view,
    )

    updater._pg_mouse_moved(object())

    assert plot_view.title == 'RGB x=1 y=0 [4, 5, 6]'
    assert plot_view.bold is True


def test_pyqtgraph_mouse_moved_grayscale_image():
    updater = _PyQtGraphUpdater(None, title='Gray')
    plot_view = _View()
    image = np.array([[1.25]], dtype=np.float32)
    updater.view = SimpleNamespace(
        imageItem=_ImageItem(image, _Point(0, 0)),
        view=plot_view,
    )

    updater._pg_mouse_moved(object())

    assert plot_view.title == 'Gray x=0 y=0 [1.25]'
    assert plot_view.bold is True
