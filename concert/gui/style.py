"""Palette-aware, locally scoped styling for Concert controls."""
from PyQt6.QtGui import QPalette


def apply_style(widget):
    """Style this widget subtree only; respect the application's light/dark palette."""
    dark = widget.palette().color(QPalette.ColorRole.Window).lightness() < 128
    colors = ({'background': '#171c26', 'card': '#222936', 'field': '#181f2b',
               'text': '#e8edf5', 'muted': '#a7b3c7', 'border': '#364155',
               'accent': '#88b4ff', 'hover': '#2d394e', 'error': '#ffabab',
               'error_bg': '#40282e', 'badge': '#293951'} if dark else
              {'background': '#f3f5f9', 'card': '#ffffff', 'field': '#f8fafc',
               'text': '#202b3c', 'muted': '#596b83', 'border': '#d7dfe9',
               'accent': '#245fc2', 'hover': '#eaf1fd', 'error': '#a52332',
               'error_bg': '#fff0f1', 'badge': '#eaf1fd'})
    widget.setStyleSheet('''
        QWidget#concertContent, QWidget#concertWindow {
            background: %(background)s; color: %(text)s;
        }
        QScrollArea#concertParameters { border: none; background: %(background)s; }
        QWidget#concertParameterRow {
            background: %(card)s; border: 1px solid %(border)s; border-radius: 10px;
        }
        QTabWidget::pane { border: none; }
        QTabBar::tab {
            color: %(muted)s; background: %(background)s; padding: 10px 18px;
            border-bottom: 2px solid transparent;
        }
        QTabBar::tab:selected { color: %(accent)s; border-bottom: 2px solid %(accent)s; }
        QLabel#concertName { color: %(text)s; font-weight: 600; background: transparent; }
        QToolButton { color: %(muted)s; background: transparent; border: none; padding: 2px; }
        QToolButton:hover { color: %(accent)s; }
        QLabel#concertUnit, QLabel#concertTarget, QLabel#concertSubtitle {
            color: %(muted)s; background: transparent;
        }
        QLabel#concertStatus {
            color: %(muted)s; background: %(field)s; border-radius: 6px; padding: 3px 7px;
        }
        QLabel#concertStatus[tone="active"] { color: %(accent)s; background: %(badge)s; }
        QLabel#concertStatus[tone="error"], QLabel#concertError {
            color: %(error)s; background: %(error_bg)s; border-radius: 6px; padding: 6px 8px;
        }
        QLineEdit#concertEditor, QComboBox#concertEditor {
            color: %(text)s; background: %(field)s; border: 1px solid %(border)s;
            border-radius: 6px; padding: 7px 9px; selection-background-color: %(accent)s;
        }
        QLineEdit#concertEditor:focus, QComboBox#concertEditor:focus {
            border: 1px solid %(accent)s;
        }
        QLineEdit#concertEditor:disabled, QComboBox#concertEditor:disabled {
            color: %(muted)s;
        }
        QLabel#concertEditor, QCheckBox#concertEditor { color: %(text)s; padding: 7px 0px; }
        QPushButton#concertAction, QPushButton#concertPrimary {
            color: %(accent)s; background: %(badge)s; border: 1px solid %(border)s;
            border-radius: 6px; padding: 7px 10px; font-weight: 600;
        }
        QPushButton#concertAction:hover, QPushButton#concertPrimary:hover {
            background: %(hover)s; border-color: %(accent)s;
        }
        QPushButton#concertAction:pressed, QPushButton#concertPrimary:pressed {
            border: 2px solid %(accent)s; padding: 6px 9px;
        }
        QPushButton#concertAction:focus, QPushButton#concertPrimary:focus {
            border-color: %(accent)s;
        }
        QPushButton#concertAction:disabled, QPushButton#concertPrimary:disabled {
            color: %(muted)s; background: %(field)s;
        }
        QLabel#concertTitle { color: %(text)s; font-size: 22px; font-weight: 600; }
        QGroupBox#concertGroup {
            color: %(text)s; border: none; font-size: 15px; font-weight: 600;
            margin-top: 24px;
        }
        QGroupBox#concertGroup::title { subcontrol-origin: margin; left: 12px; top: 4px; }
        QScrollBar:vertical { background: %(background)s; width: 10px; margin: 2px; }
        QScrollBar::handle:vertical { background: %(border)s; border-radius: 4px; min-height: 28px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
    ''' % colors)


def set_status(label, text, tone='neutral'):
    label.setText(text)
    if label.property('tone') != tone:
        label.setProperty('tone', tone)
        label.style().unpolish(label)
        label.style().polish(label)
