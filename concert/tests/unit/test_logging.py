import logging
import multiprocessing
import os
import runpy
import tempfile
import time
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import patch

from concert.session.utils import MultiprocessTimedRotatingFileHandler, setup_logging
from concert.tests import TestCase


def _write_shared_log(logfile, message, barrier):
    """Write from a separate process after all writers are ready."""
    handler = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
    try:
        barrier.wait(timeout=10)
        handler.emit(logging.LogRecord(
            name='test.logger', level=logging.INFO, pathname=__file__, lineno=0,
            msg=message, args=(), exc_info=None,
        ))
        handler.flush()
    finally:
        handler.close()


class TestLogging(TestCase):
    def setUp(self):
        super().setUp()
        logging.disable(logging.NOTSET)

    def tearDown(self):
        super().tearDown()
        root_logger = logging.getLogger()
        for handler in getattr(self, '_logging_handlers', []):
            handler.close()
            root_logger.removeHandler(handler)

    def test_setup_logging_writes_to_a_daily_rotating_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')

            setup_logging('test-session', filename=logfile, loglevel='info')
            handlers = [handler for handler in logging.getLogger().handlers
                        if isinstance(handler, TimedRotatingFileHandler)]
            self._logging_handlers = handlers

            self.assertEqual(len(handlers), 1)
            self.assertEqual(handlers[0].when, 'D')
            self.assertEqual(handlers[0].backupCount, 0)

            logging.getLogger('test.logger').info('message')
            handlers[0].flush()

            with open(logfile) as log:
                content = log.read()
            self.assertIn('test-session', content)
            self.assertIn('message', content)

    def test_daily_rotation_keeps_the_new_log_at_the_base_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            setup_logging('test-session', filename=logfile, loglevel='info')
            handler = next(handler for handler in logging.getLogger().handlers
                           if isinstance(handler, TimedRotatingFileHandler))
            self._logging_handlers = [handler]
            logging.getLogger('test.logger').info('before rotation')
            handler.flush()

            rotated = logfile + '.2026-08-05'
            with patch.object(handler, 'rotation_filename', return_value=rotated), \
                    patch('logging.handlers.time.time', return_value=handler.rolloverAt + 1):
                handler.doRollover()

            logging.getLogger('test.logger').info('after rotation')
            handler.flush()

            with open(rotated) as log:
                self.assertIn('before rotation', log.read())
            with open(logfile) as log:
                self.assertIn('after rotation', log.read())

    def test_log_follow_uses_f_for_rotating_logfiles(self):
        namespace = runpy.run_path(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bin', 'concert'),
            run_name='concert_cli',
        )
        log_command = namespace['LogCommand']()
        with patch.object(namespace['cs'], 'logfile_path', return_value='/tmp/concert.log'), \
                patch.object(namespace['cs'], 'exit_if_not_exists'), \
                patch.object(namespace['os'].path, 'exists', return_value=True), \
                patch.object(namespace['subprocess'], 'call') as call:
            log_command.run(session='session', follow=True)

        call.assert_called_once_with(
            'tail -F /tmp/concert.log | grep --line-buffered "session:"',
            shell=True,
        )


class TestMultiprocessTimedRotatingFileHandler(TestCase):
    @staticmethod
    def _record(message):
        return logging.LogRecord(
            name='test.logger', level=logging.INFO, pathname=__file__, lineno=0,
            msg=message, args=(), exc_info=None,
        )

    def test_new_handler_uses_existing_shared_rollover_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            first = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            shared_rollover = first.rolloverAt + 123
            with first._locked():
                first._write_rollover(shared_rollover)

            second = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            try:
                self.assertEqual(second.rolloverAt, shared_rollover)
            finally:
                first.close()
                second.close()

    def test_emit_recovers_from_invalid_shared_rollover_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            try:
                with handler._locked():
                    handler._rotation_lock.seek(0)
                    handler._rotation_lock.truncate()
                    handler._rotation_lock.write('invalid')
                    handler._rotation_lock.flush()

                handler.emit(self._record('message'))
                handler.flush()

                with handler._locked():
                    shared_rollover = handler._read_rollover()
                self.assertEqual(shared_rollover, handler.rolloverAt)
                with open(logfile) as log:
                    self.assertIn('message', log.read())
            finally:
                handler.close()

    def test_rollover_failure_is_reported_without_interrupting_logging_caller(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            rollover_at = time.time() - 1
            record = self._record('message during failed rollover')
            try:
                handler.emit(self._record('before rollover'))
                handler.flush()
                with handler._locked():
                    handler._write_rollover(rollover_at)

                with patch('concert.session.utils.shutil.copyfile',
                           side_effect=OSError('filesystem full')), \
                        patch.object(handler, 'handleError') as handle_error:
                    # The filesystem exception must not escape emit().
                    handler.emit(record)

                handle_error.assert_called_once_with(record)
            finally:
                handler.close()

    def test_stale_handler_does_not_repeat_rollover_or_lose_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            first = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            second = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            rollover_at = time.time() - 1
            rotated = logfile + '.rotated'
            try:
                first.emit(self._record('before rollover'))
                first.flush()
                with first._locked():
                    first._write_rollover(rollover_at)
                first.rolloverAt = rollover_at
                second.rolloverAt = rollover_at

                with patch.object(first, 'rotation_filename', return_value=rotated), \
                        patch('concert.session.utils.time.time', return_value=rollover_at + 1):
                    first.emit(self._record('first handler after rollover'))

                # This handler still has the expired deadline in memory. It must
                # observe that another handler already completed the rollover.
                with patch.object(second, 'rotation_filename') as rotation_filename:
                    second.doRollover()
                rotation_filename.assert_not_called()
                second.emit(self._record('second handler after rollover'))
                first.flush()
                second.flush()

                with open(rotated) as log:
                    rotated_content = log.read()
                with open(logfile) as log:
                    current_content = log.read()

                self.assertIn('before rollover', rotated_content)
                self.assertNotIn('after rollover', rotated_content)
                self.assertIn('first handler after rollover', current_content)
                self.assertIn('second handler after rollover', current_content)
            finally:
                first.close()
                second.close()

    def test_concurrent_processes_roll_over_once_without_losing_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            rollover_at = time.time() - 1
            try:
                handler.emit(self._record('before rollover'))
                handler.flush()
                with handler._locked():
                    handler._write_rollover(rollover_at)
            finally:
                handler.close()

            context = multiprocessing.get_context('spawn')
            barrier = context.Barrier(3)
            processes = [
                context.Process(target=_write_shared_log, args=(logfile, message, barrier))
                for message in ('process one', 'process two')
            ]
            for process in processes:
                process.start()
            try:
                barrier.wait(timeout=10)
                for process in processes:
                    process.join(timeout=10)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)

            self.assertEqual([process.exitcode for process in processes], [0, 0])
            rotated = logfile + '.' + time.strftime('%Y-%m-%d', time.localtime(rollover_at))
            archives = [
                name for name in os.listdir(tmpdir)
                if name.startswith('concert.log.') and name != 'concert.log.lock'
            ]
            self.assertEqual(archives, [os.path.basename(rotated)])
            with open(rotated) as log:
                rotated_content = log.read()
            with open(logfile) as log:
                current_content = log.read()

            self.assertIn('before rollover', rotated_content)
            self.assertNotIn('process one', rotated_content)
            self.assertNotIn('process two', rotated_content)
            self.assertIn('process one', current_content)
            self.assertIn('process two', current_content)
