from ad.session import Session
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


def test_trainable_parameters_grads():
    node1 = ad.constant(1)
    node2 = ad.variable(2)
    node3 = ad.variable(3)
    node4 = ad.variable(4)
    node5 = ad.constant(5)

    node1_node2 = ad.mul(node1, node2)
    node1_node3 = ad.mul(node1, node3)
    node1_node4 = ad.mul(node1, node4)

    node_sum = ad.add(node1_node2,
                      node1_node3,
                      node1_node4)
    final_node = ad.mul(node_sum, node5)
    trainable_nodes = final_node.trainable_parameters()

    with Session() as sess:
        out = sess.run(final_node)
    assert out.value == 45

    # test grad
    trainable_node_grads = final_node.grad(trainable_nodes)
    assert len(trainable_node_grads) == len(trainable_nodes)

    with Session() as sess:
        trainable_node_grads = sess.run(trainable_node_grads)

    for node, grad in zip(trainable_nodes, trainable_node_grads):
        print(grad.value, node.value)
        continue
        if node.name == '2':
            assert grad.value == 5
        elif node.name == '3':
            assert grad.value == 5
        elif node.name == '4':
            assert grad.value == 5
        else:
            raise ValueError('Not a trainable node')

    assert False
