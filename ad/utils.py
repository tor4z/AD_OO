from typing import Iterable


def flatten_iterable(nodes):
    outputs = []

    if not isinstance(nodes, Iterable):
        outputs.append(nodes)
    else:
        for node in nodes:
            outputs += flatten_iterable(node)

    return outputs


def find_node(target, nodes):
    for node in nodes:
        if node.__hash__ == target.__hash__:
            return True
    return False
