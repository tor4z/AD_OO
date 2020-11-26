import math
import time
from collections.abc import Iterable
from . import error


__all__ = ['Constant', 'Variable', 'Node', 'Add', 'Minus',
           'Mul', 'Neg', 'Div', 'Pow', 'Log', 'Ones', 'Zeros']


class Node(object):
    def __init__(self, name):
        """
        Bidirectional Graph
        """
        if not isinstance(name, str):
            name = str(name)

        self.value = None
        self.name = name
        self._grad = None
        self._hash = None
        self._input_nodes = []
        self._output_nodes = []
        self._grad_ref_nodes = []

    def eval(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self):
        return self.eval()

    def __eq__(self, node):
        return self.name == node.name and\
            self.value == node.value

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.name + str(time.time()))
        return self._hash

    def set_inputs(self, *nodes):
        for node in nodes:
            self.set_input(node)

    def set_outputs(self, *nodes):
        for node in nodes:
            self.set_output(node)

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

    def set_grad_refs(self, *nodes):
        for node in nodes:
            self.set_grad_ref(node)

    def set_grad_ref(self, node):
        # just for grad
        if not isinstance(node, Node):
            raise ValueError('A Node is required.')

        if node not in self._grad_ref_nodes:
            self._grad_ref_nodes.append(node)
            node.set_grad_ref(self)

    def set_root_grad(self):
        self._grad = Ones(f'Grad({self.name})')

    def grad_wrt_check(self, node):
        if node not in self._input_nodes:
            error.GradValueError(
                f'Node({node.name}) is not a input of operation {self.name}')

    def self_grad(self):
        if self._grad is None:
            if self._output_nodes:
                for node in self._output_nodes:
                    self._grad = Add(self._grad, node)
            else:
                self.set_root_grad()

    def grad(self, *wrt):
        raise NotImplementedError

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
        raise NotImplementedError


class Constant(Node):
    def __init__(self, value, name):
        if name is None:
            name = str(value)
        super().__init__(name)
        self.value = value

    def eval(self):
        return self

    def grad(self):
        raise NotImplementedError


class Ones(Constant):
    def __init__(self, name):
        super().__init__(value=1, name=name)


class Zeros(Constant):
    def __init__(self, name):
        super().__init__(value=0, name=name)


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
        raise NotImplementedError


class Operator(Node):
    def __init__(self, *input_nodes):
        name = self.generate_name(*input_nodes)
        super().__init__(name)
        self.set_inputs(*input_nodes)

    @classmethod
    def generate_name(cls, *input_nodes):
        name = f'{cls.__name__.lower()}('
        if isinstance(input_nodes, Iterable):
            for node in input_nodes:
                if node is not None:
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

    def grad(self, *wrt):
        raise NotImplementedError


class List(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        output = []
        for node in self._input_nodes:
            output.append(node.eval())
        return iter(output)

    def __iter__(self):
        return self.eval()


class Tuple(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        output = []
        for node in self._input_nodes:
            output.append(node.eval())
        return iter(output)

    def __iter__(self):
        return self.eval()


class Add(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        self.value = 0.0
        for node in self._input_nodes:
            if node is not None:
                self.value += node.eval().value
        return self

    def grad(self, *wrt):
        self.self_grad()
        output = []
        for node in wrt:
            if node in self._input_nodes:
                grad_name = f'Grad({node.name})'
                one = Mul(Ones(grad_name), self._grad)
                one.set_grad_ref(node)
                output.append(one)
            else:
                if self._input_nodes:
                    for input_node in self._input_nodes:
                        grad = input_node.grad(node)
                        output.append(grad)
                else:
                    raise error.GradValueError(
                        f'Can not found Node({node.name}) in the graph.')

        if len(output) == 1:
            return output[0]
        else:
            return Tuple(*output)


class Neg(Operator):
    def __init__(self, input_node):
        super().__init__(input_node)

    def eval(self):
        self.value = -1 * self._input_nodes[0].value
        return self

    def grad(self, *wrt):
        output = []
        for node in wrt:
            if node in self._input_nodes:
                grad_name = f'Grad({node.name})'
                one = Ones(grad_name) * self._grad * -1
                one.set_grad_ref(node)
                output.append(one)
            else:
                raise error.GradValueError(
                    f'Node({node.name}) is not a input of {self.name}')

        if len(output) == 1:
            return output[0]
        else:
            return Tuple(*output)


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

    def grad(self, *wrt):
        output = []
        for node in wrt:
            if node in self._input_nodes:
                if node == self._input_nodes[0]:
                    sign = 1
                else:
                    sign = -1

                grad_name = f'Grad({node.name})'
                one = Ones(grad_name) * self._grad * sign
                one.set_grad_ref(node)
                output.append(one)
            else:
                raise error.GradValueError(
                    f'Node({node.name}) is not a input of {self.name}')

        if len(output) == 1:
            return output[0]
        else:
            return Tuple(*output)


class Mul(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        self.value = 1.0
        for node in self._input_nodes:
            if node in self._input_nodes:
                self.value *= node.eval().value
            else:
                pass
        return self

    def grad(self, *wrt):
        output = []
        for node in wrt:
            grad_name = f'Grad({node.name})'
            output.append(Ones(grad_name))

        if len(output) == 1:
            return output[0]
        else:
            return Tuple(*output)


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
