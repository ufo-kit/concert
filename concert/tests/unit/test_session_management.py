import os
import tempfile
from unittest.mock import patch

import concert.session.management as cs
from concert.tests import TestCase


class TestSessionManagement(TestCase):
    def test_move_preserves_profile_and_remove_cleans_it_up(self):
        with tempfile.TemporaryDirectory() as root:
            source_profile = os.path.join(root, 'profile_concert_source')
            target_profile = os.path.join(root, 'profile_concert_target')
            os.mkdir(source_profile)
            with open(os.path.join(source_profile, 'history.sqlite'), 'w') as history:
                history.write('history')
            open(os.path.join(root, 'source.py'), 'w').close()
            with patch.object(cs, '_CACHED_PATH', root), \
                    patch.object(cs, 'get_ipython_dir', return_value=root):
                cs.move('source', 'target')
                self.assertFalse(os.path.exists(source_profile))
                with open(os.path.join(target_profile, 'history.sqlite')) as history:
                    self.assertEqual(history.read(), 'history')
                cs.remove('target')
                self.assertFalse(os.path.exists(target_profile))

    def test_move_rejects_existing_target_profile_without_changing_source(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, 'profile_concert_target'))
            source = os.path.join(root, 'source.py')
            open(source, 'w').close()
            with patch.object(cs, '_CACHED_PATH', root), \
                    patch.object(cs, 'get_ipython_dir', return_value=root):
                with self.assertRaises(FileExistsError):
                    cs.move('source', 'target')
            self.assertTrue(os.path.exists(source))

    def test_failed_session_move_restores_source_profile(self):
        with tempfile.TemporaryDirectory() as root:
            source_profile = os.path.join(root, 'profile_concert_source')
            os.mkdir(source_profile)
            with patch.object(cs, '_CACHED_PATH', root), \
                    patch.object(cs, 'get_ipython_dir', return_value=root):
                with self.assertRaises(FileNotFoundError):
                    cs.move('missing', 'target')
                # Exercise rollback after a profile was actually moved.
                with self.assertRaises(FileNotFoundError):
                    cs.move('source', 'target')
            self.assertTrue(os.path.exists(source_profile))
            self.assertFalse(os.path.exists(os.path.join(root, 'profile_concert_target')))

    def test_move_without_profile(self):
        with tempfile.TemporaryDirectory() as root:
            open(os.path.join(root, 'source.py'), 'w').close()
            with patch.object(cs, '_CACHED_PATH', root), \
                    patch.object(cs, 'get_ipython_dir', return_value=root):
                cs.move('source', 'target')
            self.assertTrue(os.path.exists(os.path.join(root, 'target.py')))

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
