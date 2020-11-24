import numpy as np
import ad


def test_constant():
    # int value
    a = ad.constant(1, 'a')
    assert a.value == 1
    assert a.name == 'a'

    # float value
    a = ad.constant(0.5, 'a')
    assert a.value == 0.5
    assert a.name == 'a'

    # numpy array
    np_a = np.random.rand(5)
    a = ad.constant(np_a, 'a')
    assert np.array_equal(a.value, np_a)
    assert a.name == 'a'
