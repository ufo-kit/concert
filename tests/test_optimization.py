import measure.optimization


def test_maximizer():
    m = measure.optimization.Maximizer()
    m.value = 1.0
    assert m.is_better(2.0)
    assert not m.is_better(0.0)


def test_minimizer():
    m = measure.optimization.Minimizer()
    m.value = 1.0
    assert m.is_better(0.0)
    assert not m.is_better(1.0)
