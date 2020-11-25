import ad


def test_session_default_graph():
    sess = ad.Session()
    assert sess.graph is not None


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
