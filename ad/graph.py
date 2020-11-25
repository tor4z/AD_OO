from .singleton import Singleton


class Graph(object):
    def __init__(self):
        pass


class DefaultGraph(Graph, Singleton):
    pass
