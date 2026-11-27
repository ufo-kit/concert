import asyncio
import logging
from functools import wraps
from importlib import import_module
import threading

from concert.quantities import q
from concert.config import GUI_EVENT_REFRESH_PERIOD, GUI_MAX_PROCESSING_TIME_FRACTION

LOG = logging.getLogger(__name__)

_app = None


def get_application():
    """Explicitly create or reuse the PyQt6 application (on the main thread)."""
    global _app
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Qt application must be created on the main thread')
    try:
        from PyQt6.QtWidgets import QApplication
        import_module('qasync')  # Check optional dependencies before initializing Qt.
    except ImportError as error:
        raise RuntimeError("GUI support requires installation with 'concert[gui]'") from error
    _app = QApplication.instance() or QApplication([])
    return _app


def gui_event_loop():
    """Start a GUI infrastructure task, not an abortable Concert hardware task."""
    # IPython startup executes synchronously before its asyncio loop is running.
    return asyncio.get_event_loop().create_task(_process_events(), name='concert-gui-events')


async def _process_events():
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


class GuiSession:
    """Own GUI infrastructure and controllers for one application/session.

    qasync already dispatches Qt events, so standalone owners use ``pump=False``.
    Shutdown stops reads and waits for submitted writes, without aborting hardware.
    """
    def __init__(self, pump=False):
        self.pump = pump
        self._event_task = None
        self._controllers = set()

    def start(self):
        get_application()
        if self.pump and (self._event_task is None or self._event_task.done()):
            self._event_task = gui_event_loop()
        return self

    def register(self, controller):
        """Register a controller or async-shutdown resource; return a detach callback."""
        self._controllers.add(controller)
        return lambda: self._controllers.discard(controller)

    async def shutdown(self):
        await asyncio.gather(*(controller.shutdown() for controller in tuple(self._controllers)))
        self._controllers.clear()
        if self._event_task is not None:
            self._event_task.cancel()
            await asyncio.gather(self._event_task, return_exceptions=True)
            self._event_task = None


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
