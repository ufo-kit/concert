import logging
import multiprocessing
import os
import runpy
import tempfile
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import patch
from zoneinfo import ZoneInfo

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


def _crash_during_rollover(logfile, stage):
    handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
    handler._write_rollover(time.time() - 1)
    if stage == 'copying':
        def crash_copy(source, output):
            output.write(b'partial')
            output.flush()
            os._exit(73)
        hook = patch('concert.session.utils.shutil.copyfileobj', side_effect=crash_copy)
    elif stage == 'ready':
        original = handler._write_rotation
        def crash_ready(state):
            original(state)
            if state['phase'] == 'ready':
                os._exit(73)
        hook = patch.object(handler, '_write_rotation', side_effect=crash_ready)
    elif stage == 'truncated':
        original = os.ftruncate
        def crash_truncate(fd, size):
            original(fd, size)
            os._exit(73)
        hook = patch('concert.session.utils.os.ftruncate', side_effect=crash_truncate)
    else:
        original = handler._write_rollover
        def crash_deadline(deadline):
            original(deadline)
            os._exit(73)
        hook = patch.object(handler, '_write_rollover', side_effect=crash_deadline)
    with hook:
        handler.emit(logging.LogRecord('test', logging.INFO, '', 0, 'unwritten', (), None))


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
            self.assertEqual(handlers[0].when, 'MIDNIGHT')
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
    def test_midnight_rotation_tracks_short_and_long_dst_days(self):
        timezone = ZoneInfo('America/New_York')
        for month, day, hours in ((3, 8, 23), (11, 1, 25)):
            with self.subTest(month=month), tempfile.TemporaryDirectory() as root:
                start = datetime(2026, month, day, tzinfo=timezone).timestamp()
                next_midnight = datetime(2026, month, day + 1, tzinfo=timezone).timestamp()
                self.assertEqual(next_midnight - start, hours * 3600)
                logfile = os.path.join(root, 'concert.log')
                with open(logfile, 'w') as log:
                    log.write('previous day\n')
                os.utime(logfile, (start - 1, start - 1))

                def localtime(timestamp=None):
                    return datetime.fromtimestamp(
                        start if timestamp is None else timestamp, timezone).timetuple()

                with patch('concert.session.utils.time.localtime', side_effect=localtime), \
                        patch('concert.session.utils.time.time', return_value=start):
                    handler = MultiprocessTimedRotatingFileHandler(logfile, when='midnight')
                    try:
                        self.assertEqual(handler.rolloverAt, start)
                        handler.emit(self._record('DST transition day'))
                        self.assertEqual(handler.rolloverAt, next_midnight)
                        with patch('concert.session.utils.time.time',
                                   return_value=next_midnight - 1):
                            handler.emit(self._record('before next midnight'))
                        self.assertEqual(handler.rolloverAt, next_midnight)
                        with patch('concert.session.utils.time.time', return_value=next_midnight):
                            handler.emit(self._record('next calendar day'))
                        following_midnight = datetime(
                            2026, month, day + 2, tzinfo=timezone).timestamp()
                        self.assertEqual(handler.rolloverAt, following_midnight)
                        with handler._locked():
                            self.assertEqual(handler._read_rollover(), following_midnight)
                        archives = sorted(name for name in os.listdir(root)
                                          if name.startswith('concert.log.')
                                          and name != 'concert.log.lock')
                        self.assertEqual(len(archives), 2)
                        with open(os.path.join(root, archives[0])) as log:
                            self.assertEqual(log.read(), 'previous day\n')
                        with open(os.path.join(root, archives[1])) as log:
                            self.assertEqual(log.read(),
                                             'DST transition day\nbefore next midnight\n')
                        with open(logfile) as log:
                            self.assertEqual(log.read(), 'next calendar day\n')
                    finally:
                        handler.close()

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

    def test_nonfinite_shared_deadlines_are_replaced_on_startup_and_emit(self):
        for value in ('nan', 'inf', '-inf'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmpdir:
                logfile = os.path.join(tmpdir, 'concert.log')
                with open(logfile + '.lock', 'w') as state:
                    state.write(value)
                handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
                try:
                    with handler._locked():
                        self.assertEqual(handler._read_rollover(), handler.rolloverAt)
                        handler._rotation_lock.seek(0)
                        handler._rotation_lock.truncate()
                        handler._rotation_lock.write(value)
                        handler._rotation_lock.flush()
                    handler.emit(self._record('message'))
                    with handler._locked():
                        self.assertEqual(handler._read_rollover(), handler.rolloverAt)
                    with open(logfile) as log:
                        self.assertEqual(log.read(), 'message\n')
                finally:
                    handler.close()

    def test_backward_clock_jump_keeps_writing_until_shared_deadline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
            deadline = handler.rolloverAt
            try:
                with patch('concert.session.utils.time.time', return_value=deadline - 172800), \
                        patch.object(handler, '_doRollover') as rollover:
                    handler.emit(self._record('clock moved backward'))
                rollover.assert_not_called()
                self.assertEqual(handler.rolloverAt, deadline)
                with open(logfile) as log:
                    self.assertEqual(log.read(), 'clock moved backward\n')
            finally:
                handler.close()

    def test_forward_clock_jump_rotates_once_and_schedules_future_deadline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
            now = handler.rolloverAt + 5 * 86400
            try:
                handler.emit(self._record('before clock jump'))
                with patch('concert.session.utils.time.time', return_value=now), \
                        patch.object(handler, '_doRollover', wraps=handler._doRollover) as rollover:
                    handler.emit(self._record('after clock jump'))
                    handler.emit(self._record('next record'))
                rollover.assert_called_once()
                self.assertEqual(handler.rolloverAt, now + 86400)
                with handler._locked():
                    self.assertEqual(handler._read_rollover(), handler.rolloverAt)
                with open(logfile) as log:
                    self.assertEqual(log.read(), 'after clock jump\nnext record\n')
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

                with patch('concert.session.utils.shutil.copyfileobj',
                           side_effect=OSError('filesystem full')), \
                        patch.object(handler, 'handleError') as handle_error:
                    # The filesystem exception must not escape emit().
                    handler.emit(record)

                handle_error.assert_called_once_with(record)
            finally:
                handler.close()

    def test_retry_after_shared_state_failure_preserves_existing_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
            deadline = time.time() - 1
            try:
                handler.emit(self._record('original records'))
                with handler._locked():
                    handler._write_rollover(deadline)
                with patch.object(handler, '_write_rollover', side_effect=OSError('full')), \
                        patch.object(handler, 'handleError') as error:
                    for _ in range(3):
                        handler.emit(self._record('failed rollover'))
                self.assertEqual(error.call_count, 3)
                self.assertTrue(os.path.exists(logfile + '.rotation'))
                archive = logfile + '.' + time.strftime('%Y-%m-%d_%H-%M-%SZ',
                                                       time.gmtime(deadline))
                handler.emit(self._record('retry'))
                with open(archive) as log:
                    self.assertEqual(log.read(), 'original records\n')
                self.assertFalse(os.path.exists(archive + '.1'))
                with open(logfile) as log:
                    self.assertEqual(log.read(), 'retry\n')
            finally:
                handler.close()

    def test_failed_archive_copy_removes_partial_archive_and_keeps_active_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
            try:
                handler.emit(self._record('original records'))
                with handler._locked():
                    handler._write_rollover(time.time() - 1)

                def fail_copy(source, output):
                    output.write(b'partial')
                    raise OSError('full')

                with patch('concert.session.utils.shutil.copyfileobj', side_effect=fail_copy), \
                        patch.object(handler, 'handleError'):
                    handler.emit(self._record('failed rollover'))
                self.assertEqual(sorted(os.listdir(tmpdir)), ['concert.log', 'concert.log.lock'])
                with open(logfile) as log:
                    self.assertEqual(log.read(), 'original records\nfailed rollover\n')
            finally:
                handler.close()

    def test_process_crashes_recover_without_losing_or_duplicating_rotated_records(self):
        if os.name == 'nt':
            self.skipTest('Local POSIX crash recovery is the supported test scope')
        for stage in ('copying', 'ready', 'truncated', 'deadline'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                logfile = os.path.join(root, 'concert.log')
                survivor = MultiprocessTimedRotatingFileHandler(logfile, when='D')
                try:
                    survivor.emit(self._record('original records'))
                    survivor.flush()
                    process = multiprocessing.get_context('spawn').Process(
                        target=_crash_during_rollover, args=(logfile, stage))
                    process.start()
                    try:
                        process.join(timeout=15)
                        self.assertEqual(process.exitcode, 73)
                    finally:
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=10)
                    # Test both an already-open writer and initialization recovery.
                    if stage in ('copying', 'ready'):
                        survivor.close()
                        survivor = MultiprocessTimedRotatingFileHandler(logfile, when='D')
                    survivor.emit(self._record('after recovery'))
                    archives = [name for name in os.listdir(root)
                                if name.startswith('concert.log.')
                                and name not in ('concert.log.lock', 'concert.log.rotation')]
                    self.assertEqual(len(archives), 1)
                    with open(os.path.join(root, archives[0])) as archive:
                        self.assertEqual(archive.read(), 'original records\n')
                    with open(logfile) as active:
                        self.assertEqual(active.read(), 'after recovery\n')
                    self.assertFalse(os.path.exists(logfile + '.rotation'))
                finally:
                    survivor.close()

    def test_archives_keep_private_source_permissions(self):
        if os.name == 'nt':
            self.skipTest('POSIX permission bits are not supported on Windows')
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
            try:
                os.chmod(logfile, 0o600)
                handler.emit(self._record('private records'))
                handler._write_rollover(time.time() - 1)
                handler.emit(self._record('after rollover'))
                archive = next(name for name in os.listdir(tmpdir)
                               if name not in ('concert.log', 'concert.log.lock'))
                self.assertEqual(os.stat(os.path.join(tmpdir, archive)).st_mode & 0o777, 0o600)
            finally:
                handler.close()

    def test_reused_deadline_never_overwrites_existing_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D')
            deadline = time.time() - 1
            archive = logfile + '.' + time.strftime('%Y-%m-%d_%H-%M-%SZ', time.gmtime(deadline))
            try:
                with open(archive, 'w') as log:
                    log.write('existing archive\n')
                handler.emit(self._record('new interval'))
                handler._write_rollover(deadline)
                handler.emit(self._record('after rollover'))
                with open(archive) as log:
                    self.assertEqual(log.read(), 'existing archive\n')
                with open(archive + '.1') as log:
                    self.assertEqual(log.read(), 'new interval\n')
            finally:
                handler.close()

    def test_unsupported_handler_options_are_rejected(self):
        for options in ({'delay': True}, {'backupCount': 2}, {'interval': 0}, {'interval': -1}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(ValueError, 'delay=False and backupCount=0'):
                    MultiprocessTimedRotatingFileHandler(os.path.join(tmpdir, 'log'), **options)

    def test_failed_initialization_closes_open_streams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'log')
            streams = []
            real_open = open

            def track_open(*args, **kwargs):
                stream = real_open(*args, **kwargs)
                streams.append(stream)
                return stream

            with patch('builtins.open', side_effect=track_open), \
                    patch.object(MultiprocessTimedRotatingFileHandler, '_write_rollover',
                                 side_effect=OSError('state failure')):
                with self.assertRaises(OSError):
                    MultiprocessTimedRotatingFileHandler(logfile)
            self.assertTrue(streams)
            self.assertTrue(all(stream.closed for stream in streams))

    def test_manual_rollover_acquires_and_releases_handler_lock_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = MultiprocessTimedRotatingFileHandler(os.path.join(tmpdir, 'log'))
            try:
                with patch.object(handler, 'acquire', wraps=handler.acquire) as acquire, \
                        patch.object(handler, 'release', wraps=handler.release) as release, \
                        patch.object(handler, '_doRollover', side_effect=OSError('failed')):
                    with self.assertRaises(OSError):
                        handler.doRollover()
                acquire.assert_called_once()
                release.assert_called_once()
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

    def test_daily_rollovers_on_same_local_date_preserve_both_archives(self):
        timezone = ZoneInfo('America/New_York')
        first_deadline = datetime(2026, 11, 1, 0, 30, tzinfo=timezone).timestamp()
        second_deadline = first_deadline + 86400
        self.assertEqual(
            datetime.fromtimestamp(second_deadline, timezone).isoformat(),
            '2026-11-01T23:30:00-05:00',
        )

        # Model the server timezone without modifying the process-wide TZ.
        def localtime(timestamp):
            return datetime.fromtimestamp(timestamp, timezone).timetuple()

        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, 'concert.log')
            handler = MultiprocessTimedRotatingFileHandler(logfile, when='D', backupCount=0)
            try:
                with handler._locked():
                    handler._write_rollover(first_deadline)
                with patch('concert.session.utils.time.localtime', side_effect=localtime), \
                        patch('concert.session.utils.time.time', return_value=first_deadline - 1):
                    handler.emit(self._record('first interval'))
                with patch('concert.session.utils.time.localtime', side_effect=localtime), \
                        patch('concert.session.utils.time.time', return_value=first_deadline):
                    handler.emit(self._record('second interval'))
                self.assertEqual(handler.rolloverAt, second_deadline)
                with patch('concert.session.utils.time.localtime', side_effect=localtime), \
                        patch('concert.session.utils.time.time', return_value=second_deadline):
                    handler.emit(self._record('active interval'))
                handler.flush()

                archives = sorted(
                    name for name in os.listdir(tmpdir)
                    if name.startswith('concert.log.') and name != 'concert.log.lock'
                )
                self.assertEqual(len(archives), 2)
                with open(os.path.join(tmpdir, archives[0])) as log:
                    self.assertEqual(log.read(), 'first interval\n')
                with open(os.path.join(tmpdir, archives[1])) as log:
                    self.assertEqual(log.read(), 'second interval\n')
                with open(logfile) as log:
                    self.assertEqual(log.read(), 'active interval\n')
            finally:
                handler.close()

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
            rotated = logfile + '.' + time.strftime('%Y-%m-%d_%H-%M-%SZ',
                                                    time.gmtime(rollover_at))
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
