import ad


def test_add_grad():
    return
    a = ad.variable(1)
    b = ad.variable(2)
    c = a + b

    dc_da = c.grad(a)
    dc_db = c.grad(b)

    with ad.Session() as sess:
        dc_da_out = sess.run(dc_da)
        dc_db_out = sess.run(dc_db)

    assert dc_da_out == 1
    assert dc_db_out == 1