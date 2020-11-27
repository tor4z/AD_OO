import math
import ad


def test_session_default_graph():
    sess = ad.Session()
    assert sess._graph_set is not None


def test_session_graph():
    graph = ad.Graph()
    sess = ad.Session(graph)
    assert sess.graph is not None


def test_session_add():
    a = ad.constant(1, 'a')
    b = ad.constant(1, 'b')

    assert a.value == 1
    assert b.value == 1

    c_true = ad.constant(2, 'c')
    c = ad.add(a, b)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value


def test_session_neg():
    a = ad.constant(1, 'a')

    assert a.value == 1
    c_true = ad.constant(-1, 'c')
    c = ad.neg(a)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value


def test_session_minus():
    a = ad.constant(1, 'a')
    b = ad.constant(1, 'b')

    assert a.value == 1
    assert b.value == 1

    c_true = ad.constant(0, 'c')
    c = ad.minus(a, b)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value


def test_session_mul():
    a = ad.constant(2, 'a')
    b = ad.constant(2, 'b')

    assert a.value == 2
    assert b.value == 2

    c_true = ad.constant(4, 'c')
    c = ad.mul(a, b)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value


def test_session_div():
    a = ad.constant(1, 'a')
    b = ad.constant(2, 'b')

    assert a.value == 1
    assert b.value == 2

    c_true = ad.constant(1/2, 'c')
    c = ad.div(a, b)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value


def test_session_pow():
    a = ad.constant(2, 'a')
    b = ad.constant(3, 'b')

    assert a.value == 2
    assert b.value == 3

    c_true = ad.constant(2 ** 3, 'c')
    c = ad.pow(a, b)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value


def test_session_log():
    a = ad.constant(3, 'a')

    assert a.value == 3

    c_true = ad.constant(math.log(3), 'c')
    c = ad.log(a)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value


def test_session_reciprocal():
    a = ad.constant(3, 'a')
    assert a.value == 3

    c_true = ad.constant(1 / 3, 'c')
    c = ad.reciprocal(a)   # build graph

    with ad.Session() as sess:
        # eval graph
        c_out = sess.run(c)

    assert c_out.value == c_true.value
