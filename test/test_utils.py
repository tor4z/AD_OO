import ad


def test_flatten_iterable():
    iterable = [1, 2, 3]
    flatten_iterable = ad.utils.flatten_iterable(iterable)
    assert len(flatten_iterable) == 3

    iterable = 1
    flatten_iterable = ad.utils.flatten_iterable(iterable)
    assert len(flatten_iterable) == 1

    iterable = ([1, 2, 3])
    flatten_iterable = ad.utils.flatten_iterable(iterable)
    assert len(flatten_iterable) == 3

    iterable = ([1, 2, 3], [5, 6, 7])
    flatten_iterable = ad.utils.flatten_iterable(iterable)
    assert len(flatten_iterable) == 6

    iterable = (1)
    flatten_iterable = ad.utils.flatten_iterable(iterable)
    assert len(flatten_iterable) == 1

    iterable = (1,)
    flatten_iterable = ad.utils.flatten_iterable(iterable)
    assert len(flatten_iterable) == 1

    iterable = (([1, 2, 3], ([5, 6, 7])))
    flatten_iterable = ad.utils.flatten_iterable(iterable)
    assert len(flatten_iterable) == 6

    iterable = (([1, 2, 3], ([5, 6, 7])))
    flatten_iterable = ad.utils.flatten_iterable(iterable)

    for item in [1, 2, 3, 5, 6, 7]:
        assert item in flatten_iterable

    for item in flatten_iterable:
        assert item in [1, 2, 3, 5, 6, 7]
