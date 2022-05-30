import os
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
