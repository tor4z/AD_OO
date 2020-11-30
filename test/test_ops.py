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


def test_trainable_parameters():
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

    assert len(trainable_nodes) == 3

    for node in trainable_nodes:
        assert node.value in [2, 3, 4]
