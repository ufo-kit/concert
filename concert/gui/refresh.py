"""One visibility-aware refresh clock per panel (or standalone row)."""
import math
import asyncio
import weakref

from PyQt6 import sip
from PyQt6.QtCore import QTimer

from concert.quantities import q


def interval_seconds(interval):
    """Require a finite positive Concert duration, not unitless milliseconds."""
    seconds = float(interval.to(q.s).magnitude)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Poll interval must be finite and positive')
    return seconds


class RefreshScheduler:
    def __init__(self, owner):
        self.rows = []
        self.timer = QTimer(owner)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def tick(self):
        for row in tuple(self.rows):
            row.update_visibility()
            if row.is_value_visible():
                row._update_controls()

    def stop(self):
        self.timer.stop()


class PanelLifecycle:
    """Keep write ownership after Qt deletion without retaining the panel itself."""
    def __init__(self, panel):
        self.panel = weakref.ref(panel)
        self.controllers = [row.controller for row in panel.params.values()]

    async def shutdown(self):
        panel = self.panel()
        if panel is not None and not sip.isdeleted(panel):
            await panel.shutdown()
        else:
            await asyncio.gather(*(controller.shutdown() for controller in self.controllers))
