from .graph import Graph


class Session(object):
    def __init__(self, graph=None):
        self.graph = graph or Graph()

    def run(self, operation):
        pass
