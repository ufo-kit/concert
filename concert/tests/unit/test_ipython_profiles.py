import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from concert.tests import TestCase


class TestIPythonProfiles(TestCase):
    def test_system_command_creates_and_removes_profile(self):
        concert_command = shutil.which('concert')
        if concert_command is None:
            self.skipTest('concert command is not installed')

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = os.environ.copy()
            env.pop('VIRTUAL_ENV', None)
            env['IPYTHONDIR'] = str(root / 'ipython')
            env['XDG_DATA_HOME'] = str(root / 'data')

            self._run_concert(env, 'init', 'system', executable=concert_command)
            self._run_concert(
                env,
                'start',
                'system',
                input_text='system_marker = 1\nexit()\n',
                executable=concert_command,
            )

            profile = root / 'ipython' / 'profile_concert_system'
            self.assertTrue(profile.is_dir())
            self.assertTrue((profile / 'history.sqlite').is_file())

            self._run_concert(env, 'rm', 'system', executable=concert_command)
            self.assertFalse(profile.exists())

    def test_sessions_have_separate_histories_and_rm_removes_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = os.environ.copy()
            env.pop('VIRTUAL_ENV', None)
            env['IPYTHONDIR'] = str(root / 'ipython')
            env['XDG_DATA_HOME'] = str(root / 'data')

            for session in ('first', 'second'):
                self._run_concert(env, 'init', session)

            self._run_concert(env, 'start', 'first', input_text="first_marker = 1\nexit()\n")
            self._run_concert(env, 'start', 'second', input_text="second_marker = 2\nexit()\n")

            first_history = root / 'ipython' / 'profile_concert_first' / 'history.sqlite'
            second_history = root / 'ipython' / 'profile_concert_second' / 'history.sqlite'
            self.assertTrue(first_history.is_file())
            self.assertTrue(second_history.is_file())
            self.assertNotEqual(first_history, second_history)
            self.assertIn('first_marker', self._history(first_history))
            self.assertNotIn('second_marker', self._history(first_history))
            self.assertIn('second_marker', self._history(second_history))
            self.assertNotIn('first_marker', self._history(second_history))

            self._run_concert(env, 'rm', 'first')
            self.assertFalse(first_history.parent.exists())
            self.assertTrue(second_history.parent.exists())

    def test_filename_session_uses_its_own_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_file = root / 'external_session.py'
            session_file.write_text('external_marker = 1\n')
            env = os.environ.copy()
            env.pop('VIRTUAL_ENV', None)
            env['IPYTHONDIR'] = str(root / 'ipython')
            env['XDG_DATA_HOME'] = str(root / 'data')

            self._run_concert(env, 'start', '--filename', str(session_file),
                              input_text="exit()\n")

            history = root / 'ipython' / 'profile_concert_external_session' / 'history.sqlite'
            self.assertTrue(history.is_file())

    @staticmethod
    def _run_concert(env, *arguments, input_text=None, executable=None):
        command = executable or sys.executable
        prefix = [command] if executable else [command, 'bin/concert']
        result = subprocess.run(
            [*prefix, *arguments],
            env=env,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise AssertionError(
                f"concert {' '.join(arguments)} failed:\n{result.stdout}\n{result.stderr}"
            )
        return result

    @staticmethod
    def _history(filename):
        with sqlite3.connect(filename) as database:
            return '\n'.join(row[0] for row in database.execute('SELECT source FROM history'))
