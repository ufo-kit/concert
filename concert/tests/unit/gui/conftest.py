"""Dedicated Qt tests: choose the binding before pytest-qt creates QApplication."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_API', 'pyqt6')
os.environ.setdefault('PYTEST_QT_API', 'pyqt6')
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt6')
