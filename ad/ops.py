from collections.abc import Iterable
from . import error


__all__ = ['Constant', 'Variable', 'Add', 'Mul', 'Node']


class Node(object):
    def __init__(self, name):
        """
        Bidirectional Graph
        """
        if not isinstance(name, str):
            name = str(name)

        self.value = None
        self.name = name
        self._input_nodes = []
        self._output_nodes = []

    def eval(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self):
        return self.eval()

    def set_inputs(self, nodes):
        if isinstance(nodes, Iterable):
            for node in nodes:
                self.set_input(node)

    def set_input(self, node):
        if not isinstance(node, Node):
            raise ValueError('A Node is required.')

        if node not in self._input_nodes:
            self._input_nodes.append(node)
            node.set_output(self)

    def set_output(self, node):
        if not isinstance(node, Node):
            raise ValueError('A Node is required.')

        if node not in self._output_nodes:
            self._output_nodes.append(node)
            node.set_input(self)

    def __str__(self):
        return f'<Node({self.name})>'

    __repr__ = __str__

    def clean(self):
        del self._input_nodes
        del self._output_nodes
        self._input_nodes = []
        self._output_nodes = []


class Variable(Node):
    def __init__(self, value, name):
        if name is None:
            name = str(value)
        super().__init__(name)
        self.value = value

    def eval(self):
        return self

    def grad(self):
        pass


class Constant(Node):
    def __init__(self, value, name):
        if name is None:
            name = str(value)
        super().__init__(name)
        self.value = value

    def eval(self):
        return self

    def grad(self):
        pass


class PlaceHolder(Node):
    def __init__(self, name):
        self.value = None
        super().__init__(name)

    def eval(self):
        if self.value is None:
            raise error.PlaceholderValueError(
                f'{self.name}\'s value not set.')
        return self

    def feed_value(self, feed_dict):
        try:
            self.value = feed_dict[self.name]
        except KeyError:
            raise error.PlaceholderValueError(
                f'Key {self.name} not found in feed_dict.')

    def grad(self):
        pass


class Op(Node):
    def __init__(self, name=None):
        super().__init__(name)

    def eval(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.eval()


class Add(Op):
    def __init__(self, *input_nodes):
        name = 'add('
        for node in input_nodes:
            name += node.name + ','
        name = name[:-1]    # drop ','
        name += ')'

        super().__init__(name)
        self.set_inputs(input_nodes)

    def eval(self):
        self.value = 0.0
        for node in self._input_nodes:
            self.value += node.eval().value
        return self


class Mul(Op):
    def __init__(self):
        pass
