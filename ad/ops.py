import math
from collections.abc import Iterable
from . import error


__all__ = ['Constant', 'Variable', 'Node', 'Add', 'Minus',
           'Mul', 'Neg', 'Div', 'Pow', 'Log']


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
        else:
            self.set_input(nodes)

    def set_outputs(self, nodes):
        if isinstance(nodes, Iterable):
            for node in nodes:
                self.set_output(node)
        else:
            self.set_output(nodes)

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

    def grad(self):
        pass

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


class Operator(Node):
    def __init__(self, *input_nodes):
        name = self.generate_name(*input_nodes)
        super().__init__(name)
        self.set_inputs(input_nodes)

    @classmethod
    def generate_name(cls, *input_nodes):
        name = f'{cls.__name__.lower()}('
        if isinstance(input_nodes, Iterable):
            for node in input_nodes:
                name += node.name + ','
            name = name[:-1]    # drop ','
        else:
            name += input_nodes.name
        name += ')'
        return name

    def eval(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.eval()

    def grad(self):
        pass


class Add(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        self.value = 0.0
        for node in self._input_nodes:
            self.value += node.eval().value
        return self

    def grad(self):
        pass


class Neg(Operator):
    def __init__(self, input_node):
        super().__init__(input_node)

    def eval(self):
        self.value = -1 * self._input_nodes[0].value
        return self

    def grad(self):
        pass


class Minus(Operator):
    def __init__(self, *input_nodes):
        if len(input_nodes) != 2:
            raise ValueError('Minus accept 2 arguments.')
        super().__init__(*input_nodes)

    def eval(self):
        first_node = self._input_nodes[0]
        second_node = self._input_nodes[1]
        self.value = first_node.value - second_node.value
        return self

    def grad(self):
        pass


class Mul(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        self.value = 1.0
        for node in self._input_nodes:
            self.value *= node.eval().value
        return self

    def grad(self):
        pass


class Div(Operator):
    def __init__(self, *input_nodes):
        if len(input_nodes) != 2:
            raise ValueError('Div accept 2 arguments.')
        super().__init__(*input_nodes)

    def eval(self):
        first_node = self._input_nodes[0]
        second_node = self._input_nodes[1]

        if second_node.value == 0:
            raise ValueError('Denominator is 0')

        self.value = first_node.value / second_node.value
        return self

    def grad(self):
        pass


class Pow(Operator):
    def __init__(self, *input_nodes):
        if len(input_nodes) != 2:
            raise ValueError('Power accept 2 arguments.')
        super().__init__(*input_nodes)

    def eval(self):
        first_node = self._input_nodes[0]
        second_node = self._input_nodes[1]

        self.value = first_node.value ** second_node.value
        return self

    def grad(self):
        pass


class Log(Operator):
    def __init__(self, input_node):
        super().__init__(input_node)

    def eval(self):
        node = self._input_nodes[0]
        if node.value <= 0:
            raise ValueError('Negtive value for log.')
        self.value = math.log(node.value)

    def grad(self):
        pass
