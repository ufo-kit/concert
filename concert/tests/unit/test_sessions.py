"""Test sessions."""
import time
from concert.quantities import q
from concert.session.utils import check_emergency_stop
from concert.tests import suppressed_logging, slow


@slow
@suppressed_logging
def test_check_emergency_stop():
    class Callable(object):
        abort = True
        num_called = 0

        def __call__(self):
            was = self.abort
            if self.num_called < 2:
                # Called enough times to abort and clear
                self.abort = not was

            self.num_called += 1

            return was

    clb = Callable()
    check_emergency_stop(clb, poll_interval=1*q.ms)
    slept = 0
    poll_interval = .1

    while clb.num_called < 2:
        time.sleep(poll_interval)
        slept += poll_interval
        # Don't get stuck
        assert slept < 1

    # We aborted and cleared, thus are ready to abort again.
    assert clb.abort
