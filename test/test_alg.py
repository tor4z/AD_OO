import ad
from ad import algorithms
import time


def test_new_node(name=None):
    node = ad.Node(name or time.time())
    assert node is not None
    return node


def test_build_graph():
    #     2
    #   /   \
    # 1       4
    #   \   /
    #     3
    node1 = test_new_node(1)
    node2 = test_new_node(2)
    node3 = test_new_node(3)
    node4 = test_new_node(4)

    node1.set_output(node2)
    node1.set_output(node3)
    node4.set_input(node2)
    node4.set_input(node3)

    assert len(node1._output_nodes) == 2
    assert len(node1._input_nodes) == 0

    assert len(node2._output_nodes) == 1
    assert len(node2._input_nodes) == 1

    assert len(node3._output_nodes) == 1
    assert len(node3._input_nodes) == 1

    assert len(node4._output_nodes) == 0
    assert len(node4._input_nodes) == 2

    return node4


def test_dfs():
    visited = []
    root = test_build_graph()
    algorithms.dfs(root, visited)

    assert len(visited) == 4


def test_topsort():
    root = test_build_graph()
    outs = algorithms.topsort(root)

    assert len(outs) == 4
    for i, node in enumerate(outs):
        if i == 0:
            # first node is '1'
            assert node.name == '1'

        if i == 1 or i == 2:
            # 1st or 2nd node is '2' or '3'
            assert node.name == '2' or node.name == '3'

        if i == 3:
            # latest node is '4'
            assert node.name == '4'
