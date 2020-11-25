from .graph import DefaultGraph
from .algorithms import topsort


_default_graph = DefaultGraph()


class Session(object):
    def __init__(self, graph=None):
        if graph is not None:
            self.graph = graph
        else:
            self.graph = _default_graph

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def run(self, operation, feed_dict={}):
        sorted_nodes = topsort(operation)
        out_node = None

        for node in sorted_nodes:
            node.eval()
            out_node = node

        return out_node
