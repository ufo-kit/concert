import timeout_decorator
from concert.tests import TestCase
from concert.experiments.base import Experiment
from concert.storage import DummyWalker


class TestIssue474(TestCase):

    @timeout_decorator.timeout(1)
    def test_non_formattable_name_fmt(self):
        walker = DummyWalker()
        walker.descend('foo')
        with self.assertRaises(ValueError):
            ex = Experiment([], walker=walker, separate_scans=True, name_fmt='foo')
