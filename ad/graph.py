from . import error
from .singleton import Singleton
from .algorithms import topsort


class Graph(object):
    def __init__(self):
        self._sorted_nodes = []
        self._root = None

    @property
    def root(self):
        if self._root is None:
            raise error.GraphError('root not set.')
        return self._root

    @root.setter
    def root(self, root):
        self.set_root(root)

    def set_root(self, root):
        self._root = root

    @property
    def sorted_nodes(self):
        if not self._sorted_nodes:
            self._sorted_nodes = topsort(self.root)
        return self._sorted_nodes


class GraphSet(object):
    def __init__(self):
        self._graphs = {}
        self._current_root = None

    @property
    def current(self):
        # return current graph
        if self._current_root is None:
            raise error.GradValueError('Current root not set.')
        return self._graphs[self._current_root]

    @staticmethod
    def new_graph(root):
        graph = Graph()
        graph.root = root
        return graph

    def find_set_graph(self, root):
        if root is None:
            raise error.GraphError('root is None.')
        self._current_root = root
        if root not in self._graphs.keys():
            graph = self.new_graph(root)
            self._graphs[root] = graph
        return self.current


class DefaultGraphSet(GraphSet, Singleton):
    pass
