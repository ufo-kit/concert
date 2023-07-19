import multiprocessing as mp
import numpy as np
from concert.ext.viewers import _start_command
from concert.tests import suppressed_logging


_MP_CTX = mp.get_context('spawn')


@suppressed_logging
def test_start_command():
    image = np.arange(25).reshape(5, 5).astype(np.float32)
    proc = _MP_CTX.Process(target=_start_command, args=("echo", image), daemon=False)
    proc.start()
    proc.join()
