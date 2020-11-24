import ad


def test_constant():
    a = ad.constant(1, 'a')
    assert a.value == 1
    assert a.name == 'a'
