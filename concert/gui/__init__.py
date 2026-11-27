import asyncio
import logging
from functools import wraps
from importlib import import_module

from concert.quantities import q
from concert.config import GUI_EVENT_REFRESH_PERIOD, GUI_MAX_PROCESSING_TIME_FRACTION
from concert.coroutines.base import background

LOG = logging.getLogger(__name__)

_app = None


def get_application():
    """Explicitly create or reuse the PyQt6 application (on the main thread)."""
    global _app
    try:
        from PyQt6.QtWidgets import QApplication
        import_module('qasync')  # Check optional dependencies before initializing Qt.
    except ImportError as error:
        raise RuntimeError("GUI support requires installation with 'concert[gui]'") from error
    _app = QApplication.instance() or QApplication([])
    return _app


@background
async def gui_event_loop():
    from PyQt6.QtCore import QEventLoop
    app = get_application()
    if GUI_EVENT_REFRESH_PERIOD.magnitude <= 0 or not 0 < GUI_MAX_PROCESSING_TIME_FRACTION < 1:
        raise ValueError('GUI refresh period must be positive and processing fraction between 0 and 1')
    concert_time = GUI_EVENT_REFRESH_PERIOD * (1. - GUI_MAX_PROCESSING_TIME_FRACTION)
    gui_time = GUI_EVENT_REFRESH_PERIOD * GUI_MAX_PROCESSING_TIME_FRACTION
    while True:
        try:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,
                              int(gui_time.to(q.ms).magnitude))
        except Exception as e:
            LOG.error(e)
        await asyncio.sleep(concert_time.to(q.s).magnitude)


def with_signals(start_signal=None, finish_signal=None, exception_signal=None):
    def signal_wrapper(coro):
        @wraps(coro)
        async def wrapped(*args, **kwargs):
            if start_signal is not None:
                start_signal.emit()
            try:
                return await coro(*args, **kwargs)
            except Exception as error:
                if exception_signal is not None:
                    exception_signal.emit(error)
                raise
            finally:
                if finish_signal is not None:
                    finish_signal.emit()
        return wrapped
    return signal_wrapper
