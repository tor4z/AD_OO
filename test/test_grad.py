import ad
import math


def test_add_grad():
    a = ad.variable(1)
    b = ad.variable(2)
    c = ad.add(a, b)

    dc_da = c.grad(a)
    dc_db = c.grad(b)

    with ad.Session() as sess:
        dc_da_out = sess.run(dc_da)
        dc_db_out = sess.run(dc_db)

    assert dc_da_out.value == 1
    assert dc_db_out.value == 1


def test_mul_grad():
    a = ad.variable(1)
    b = ad.variable(2)
    c = ad.mul(a, b)

    dc_da = c.grad(a)
    dc_db = c.grad(b)

    with ad.Session() as sess:
        dc_da_out = sess.run(dc_da)
        dc_db_out = sess.run(dc_db)

    assert dc_da_out.value == 2
    assert dc_db_out.value == 1


def test_div_grad():
    a = ad.variable(1)
    b = ad.variable(2)
    c = ad.div(a, b)

    dc_da = c.grad(a)
    dc_db = c.grad(b)

    with ad.Session() as sess:
        dc_da_out = sess.run(dc_da)
        dc_db_out = sess.run(dc_db)

    assert dc_da_out.value == 1 / 2
    assert dc_db_out.value == -1 / 4


def test_pow_grad():
    a = ad.variable(2)
    b = ad.variable(3)
    c = ad.pow(a, b)

    dc_da = c.grad(a)
    dc_db = c.grad(b)

    with ad.Session() as sess:
        dc_da_out = sess.run(dc_da)
        dc_db_out = sess.run(dc_db)

    assert dc_da_out.value == 3 * 2 ** 2
    assert dc_db_out.value == 2 ** 3 * math.log(2)


def test_log_grad():
    a = ad.variable(2)
    c = ad.log(a)

    dc_da = c.grad(a)

    with ad.Session() as sess:
        dc_da_out = sess.run(dc_da)

    assert dc_da_out.value == 1 / 2


def test_reciprocal_grad():
    a = ad.variable(2)
    c = ad.reciprocal(a)

    dc_da = c.grad(a)

    with ad.Session() as sess:
        dc_da_out = sess.run(dc_da)

    assert dc_da_out.value == -1 / 4
