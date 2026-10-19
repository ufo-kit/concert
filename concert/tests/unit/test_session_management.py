import os
import tempfile
from unittest.mock import patch

import concert.session.management as cs
from concert.tests import TestCase


class TestSessionManagement(TestCase):
    def test_get_docstring(self):
        filename = os.path.join(
            os.getcwd(),
            'concert',
            'tests',
            'util',
            '_aimport_future_imports.py'
        )
        self.assertEqual(cs.get_docstring(filename), 'docstring')

    def test_remove_deletes_session_and_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = os.path.join(tmpdir, 'session.py')
            profile_path = os.path.join(tmpdir, 'profile_concert_session')
            os.mkdir(profile_path)
            open(session_path, 'w').close()

            with patch.object(cs, '_CACHED_PATH', tmpdir), \
                    patch.object(cs, 'get_ipython_dir', return_value=tmpdir):
                cs.remove('session')

            self.assertFalse(os.path.exists(session_path))
            self.assertFalse(os.path.exists(profile_path))

    def test_remove_without_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = os.path.join(tmpdir, 'session.py')
            open(session_path, 'w').close()

            with patch.object(cs, '_CACHED_PATH', tmpdir), \
                    patch.object(cs, 'get_ipython_dir', return_value=tmpdir):
                cs.remove('session')

            self.assertFalse(os.path.exists(session_path))

    def test_remove_deletes_orphaned_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, 'profile_concert_session')
            os.mkdir(profile_path)

            with patch.object(cs, '_CACHED_PATH', tmpdir), \
                    patch.object(cs, 'get_ipython_dir', return_value=tmpdir):
                cs.remove('session')

            self.assertFalse(os.path.exists(profile_path))
