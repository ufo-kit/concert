import os
import sys
import warnings
from concert._aimport import AsyncMetaPathFinder
from concert.tests import TestCase


class TestAImport(TestCase):
    def setUp(self):
        self.import_path = os.path.join(os.getcwd(), 'concert', 'tests', 'util')
        self.async_path_finder = AsyncMetaPathFinder()
        sys.meta_path.insert(0, self.async_path_finder)
        sys.path.insert(0, self.import_path)

    def tearDown(self):
        # Do not rely on the index in setUp
        sys.path.remove(self.import_path)
        sys.meta_path.remove(self.async_path_finder)

    def test_nested_aimport(self):
        self._check_values()

    def test_import_in_thread(self):
        import threading

        def import_in_thread():
            self._check_values()

        t = threading.Thread(target=import_in_thread)
        t.start()
        t.join()

    async def test_mixed_loop_await(self):
        with warnings.catch_warnings():
            # Otherwise we'd get "RuntimeWarning: coroutine 'sleep' was never awaited"
            # Also, the test needs to be `async def' for the warning to be suppressed
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with self.assertRaises(ImportError):
                import _mixed_loop_await_session
                # Prevent flake8 F401: imported but unused
                _mixed_loop_await_session

    def test_aimport_in_async_func(self):
        with warnings.catch_warnings():
            # Otherwise we'd get "RuntimeWarning: coroutine 'sleep' was never awaited"
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with self.assertRaises(ImportError):
                import _aimport_in_async_func_session
                # Prevent flake8 F401: imported but unused
                _aimport_in_async_func_session

    def _check_values(self):
        from _session import value, nested_value
        self.assertEqual(value, 0)
        self.assertEqual(nested_value, 0)
