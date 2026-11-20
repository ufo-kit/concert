import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from concert.tests import TestCase


class TestIPythonProfiles(TestCase):
    def test_checkout_command_creates_and_removes_profile_from_other_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = os.environ.copy()
            env.pop('VIRTUAL_ENV', None)
            env['IPYTHONDIR'] = str(root / 'ipython')
            env['XDG_DATA_HOME'] = str(root / 'data')

            self._run_concert(env, 'init', 'system', cwd=tmpdir)
            self._run_concert(
                env,
                'start',
                'system',
                input_text='system_marker = 1\nexit()\n',
                cwd=tmpdir,
            )

            profile = root / 'ipython' / 'profile_concert_system'
            self.assertTrue(profile.is_dir())
            self.assertTrue((profile / 'history.sqlite').is_file())

            self._run_concert(env, 'rm', 'system', cwd=tmpdir)
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

    def test_renaming_session_preserves_history_and_removal_deletes_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = os.environ.copy()
            env.pop('VIRTUAL_ENV', None)
            env['IPYTHONDIR'] = str(root / 'ipython')
            env['XDG_DATA_HOME'] = str(root / 'data')
            self._run_concert(env, 'init', 'original')
            self._run_concert(env, 'start', 'original', input_text='original_marker = 1\nexit()\n')
            self._run_concert(env, 'mv', 'original', 'renamed')
            self._run_concert(env, 'start', 'renamed', input_text='renamed_marker = 2\nexit()\n')
            profile = root / 'ipython' / 'profile_concert_renamed'
            history = self._history(profile / 'history.sqlite')
            self.assertIn('original_marker', history)
            self.assertIn('renamed_marker', history)
            self.assertFalse((root / 'ipython' / 'profile_concert_original').exists())
            self._run_concert(env, 'rm', 'renamed')
            self.assertFalse(profile.exists())

    @staticmethod
    def _run_concert(env, *arguments, input_text=None, cwd=None):
        repository = Path(__file__).resolve().parents[3]
        env = env.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            [str(repository), env['PYTHONPATH']] if env.get('PYTHONPATH') else [str(repository)]
        )
        result = subprocess.run(
            [sys.executable, str(repository / 'bin' / 'concert'), *arguments],
            env=env,
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
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
