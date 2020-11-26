from .graph import DefaultGraphSet


_default_graph_set = DefaultGraphSet()


class Session(object):
    def __init__(self, graph=None):
        self._current_root = None
        self._graph = None
        self._graph_set = None
        if graph is not None:
            self._graph = graph
        else:
            self._graph_set = _default_graph_set

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @property
    def graph(self):
        if self._graph is not None:
            return self._graph
        else:
            return self._graph_set.find_set_graph(
                self._current_root)

    def run(self, operation, feed_dict={}):
        self._current_root = operation
        out_node = None

        for node in self.graph.sorted_nodes:
            node.eval()
            out_node = node

        return out_node
