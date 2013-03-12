from concert.feedback.optimization import Maximizer, Minimizer


def test_maximizer():
    m = Maximizer()
    m.value = 1.0
    assert m.is_better(2.0)
    assert not m.is_better(0.0)


def test_minimizer():
    m = Minimizer()
    m.value = 1.0
    assert m.is_better(0.0)
    assert not m.is_better(1.0)
