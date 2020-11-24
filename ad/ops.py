from collections.abc import Iterable


class Node(object):
    def __init__(self, name):
        """
        Bidirectional Graph
        """
        self.name = name
        self._input_nodes = []
        self._output_nodes = []

    def eval(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self):
        return self.eval()

    def set_inputs(self, nodes):
        if isinstance(nodes, Iterable):
            pass

    def set_input(self, node):
        self._input_nodes.append(node)

    def set_output(self, node):
        self._output_nodes.append(node)

    def clean(self):
        del self._input_nodes
        del self._output_nodes
        self._input_nodes = []
        self._output_nodes = []


class Variable(Node):
    def __init__(self):
        pass


class Constant(Node):
    def __init__(self, value, name):
        self.value = value
        if name is None:
            name = str(value)
        super().__init__(name)

    def eval(self):
        return self

    def grad(self):
        pass


class PlaceHolder(Node):
    def __init__(self):
        self.value = None

    def eval(self):
        pass

    def set_value(self, value):
        pass


class Op(Node):
    def __init__(self):
        pass

    def eval(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.eval()


class Add(Op):
    def __init__(self, *input_nodes):
        pass

    def eval(self):
        pass


class Mul(Op):
    def __init__(self):
        pass
