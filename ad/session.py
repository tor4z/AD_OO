from .graph import Graph


class Session(object):
    def __init__(self, graph=None):
        self.graph = graph or Graph()

    def __entry__(self):
        pass

    def __exit__(self):
        pass

    def run(self, operation):
        pass
