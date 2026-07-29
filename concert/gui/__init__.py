import asyncio
import logging
import pyqtgraph as pg
from functools import wraps
from PyQt5.QtCore import QEventLoop

from concert.quantities import q
from concert.config import GUI_EVENT_REFRESH_PERIOD, GUI_MAX_PROCESSING_TIME_FRACTION
from concert.coroutines.base import background

from qasync import asyncSlot

LOG = logging.getLogger(__name__)

app = pg.mkQApp()


@background
async def gui_event_loop():
    concert_time = GUI_EVENT_REFRESH_PERIOD * (1. - GUI_MAX_PROCESSING_TIME_FRACTION)
    gui_time = GUI_EVENT_REFRESH_PERIOD * GUI_MAX_PROCESSING_TIME_FRACTION
    while True:
        try:
            app.processEvents(QEventLoop.AllEvents, int(gui_time.to(q.ms).magnitude))
        except Exception as e:
            LOG.error(e)
        await asyncio.sleep(concert_time.to(q.s).magnitude)


def with_signals(start_signal=None, finish_signal=None, exception_signal=None):
    def signal_wrapper(coro):
        if start_signal:
            start_signal.emit()
        try:
            coro()
        except Exception as e:
            if exception_signal:
                exception_signal.emit(e)
        finally:
            if finish_signal:
                finish_signal.emit()
    return signal_wrapper
