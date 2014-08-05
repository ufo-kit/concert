from __future__ import absolute_import
import gevent
from IPython.lib.inputhook import stdin_ready


def inputhook_gevent():
    try:
        while not stdin_ready():
            gevent.sleep(0.05)
    except KeyboardInterrupt:
        pass

    return 0
