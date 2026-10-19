import logging
import os
import runpy
import tempfile
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import patch

from concert.session.utils import setup_logging
from concert.tests import TestCase


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
