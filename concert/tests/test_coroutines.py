from concert.helpers import inject
from concert.coroutines import null
from concert.tests import TestCase


class TestCoroutines(TestCase):

    def setUp(self):
        super(TestCoroutines, self).setUp()
        self.iteration = 0
        self.num_iterations = 10

    def producer(self):
        for i in range(self.num_iterations):
            yield i
            self.iteration = i

    def test_null(self):
        inject(self.producer(), null())
        self.assertEquals(self.iteration, self.num_iterations - 1)
