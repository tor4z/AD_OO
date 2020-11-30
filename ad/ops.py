import math
import time
from collections.abc import Iterable
from . import algorithms
from . import error
from .utils import flatten_iterable, find_node


__all__ = ['Constant', 'Variable', 'Node', 'Add', 'Minus',
           'Mul', 'Neg', 'Div', 'Pow', 'Log', 'Reciprocal',
           'Ones', 'Zeros']


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
        return self.__hash__ == node.__hash__

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
                    # Fix me
                    if self._grad is None:
                        self._grad = node
                    else:
                        self._grad = Add(self._grad, node)
            else:
                self.set_root_grad()

    def eval_grad(self, wrt):
        raise NotImplementedError

    def _node_grad(self, wrt):
        self.self_grad()

        # dfs search wrt node
        if find_node(wrt, self._input_nodes):
            # Fix me
            grad_node = self.eval_grad(wrt)
            return grad_node
        else:
            if self._input_nodes:
                for input_node in self._input_nodes:
                    grad_node = input_node._node_grad(wrt)
                    if grad_node is not None:
                        return grad_node
            else:
                return None

    def grad(self, *wrt):
        wrt = flatten_iterable(wrt)
        output = []

        for node in wrt:
            grad_node = self._node_grad(node)
            if grad_node is not None:
                output.append(grad_node)

        if len(output) == 1:
            return output[0]
        else:
            return Tuple(*output)

    def trainable_parameters(self):
        # current node as root
        all_node = []
        all_trainable_node = []
        algorithms.dfs(self, all_node)

        for node in all_node:
            if isinstance(node, Variable):
                all_trainable_node.append(node)
        return all_trainable_node

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


class Constant(Node):
    def __init__(self, value, name):
        if name is None:
            name = str(value)
        super().__init__(name)
        self.value = value

    def eval(self):
        return self


class Ones(Constant):
    def __init__(self, name=None):
        super().__init__(value=1, name=name)


class Zeros(Constant):
    def __init__(self, name=None):
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


class List(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        output = []
        for node in self._input_nodes:
            output.append(node.eval())
        return iter(output)

    def __len__(self):
        return len(self._input_nodes)

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

    def __len__(self):
        return len(self._input_nodes)

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

    def eval_grad(self, wrt):
        grad_name = f'Grad({wrt.name})'
        grad_node = Mul(Ones(grad_name), self._grad)
        grad_node.set_grad_ref(wrt)
        return grad_node


class Neg(Operator):
    def __init__(self, input_node):
        super().__init__(input_node)

    def eval(self):
        self.value = -1 * self._input_nodes[0].value
        return self

    def eval_grad(self, wrt):
        grad_name = f'Grad({wrt.name})'
        neg_sign = Constant(-1, 'Neg_sgn')
        grad_node = Mul(Ones(grad_name), self._grad, neg_sign)
        grad_node.set_grad_ref(wrt)
        return grad_node


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

    def eval_grad(self, wrt):
        grad_name = f'Grad({wrt.name})'
        if wrt == self._input_nodes[0]:
            sign = Constant(1, 'sign')
        else:
            sign = Constant(-1, 'sign')
        grad_node = Mul(Ones(grad_name), self._grad, sign)
        grad_node.set_grad_ref(wrt)
        return grad_node


class Mul(Operator):
    def __init__(self, *input_nodes):
        super().__init__(*input_nodes)

    def eval(self):
        self.value = 1.0
        for node in self._input_nodes:
            self.value *= node.eval().value
        return self

    def eval_grad(self, wrt):
        grad_name = f'Grad({wrt.name})'
        grad_node = Ones(grad_name)
        for node in self._input_nodes:
            if node != wrt:
                grad_node = Mul(grad_node, node)

        grad_node = Mul(grad_node, self._grad)
        grad_node.set_grad_ref(wrt)
        return grad_node


class Div(Operator):
    def __init__(self, *input_nodes):
        if len(input_nodes) != 2:
            raise ValueError('Div accept 2 arguments.')
        super().__init__(*input_nodes)

    def eval(self):
        numerator_node = self._input_nodes[0]
        denominator_node = self._input_nodes[1]

        if denominator_node.value == 0:
            raise ValueError('Denominator is 0')

        self.value = numerator_node.value / denominator_node.value
        return self

    def eval_grad(self, wrt):
        grad_name = f'Grad({wrt.name})'

        numerator_node = self._input_nodes[0]
        denominator_node = self._input_nodes[1]

        if wrt == numerator_node:
            # Fix me
            grad_node = Constant(1 / denominator_node.value, grad_name)
        elif wrt == denominator_node:
            grad_node = Constant(numerator_node.value, grad_name)
            grad_node = Mul(grad_node, Constant(-1, 'sign'))
            grad_node = Div(
                grad_node, Pow(denominator_node, Constant(2, 'pow')))
        else:
            raise error.GradValueError(
                f'{wrt.name} is neither numerator nor denominator')

        grad_node = Mul(grad_node, self._grad)
        grad_node.set_grad_ref(wrt)
        return grad_node


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

    def eval_grad(self, wrt):
        grad_name = f'Grad({wrt.name})'
        base_node = self._input_nodes[0]
        power_node = self._input_nodes[1]

        if wrt == base_node:
            grad_node = Mul(
                power_node, Pow(base_node, Minus(power_node, Ones())))
        elif wrt == power_node:
            grad_node = self
            grad_node.name = grad_name
            grad_node = Mul(grad_node, Log(base_node))
        else:
            raise error.GradValueError(
                f'{wrt.name} is neither base nor power')

        grad_node = Mul(grad_node, self._grad)
        grad_node.set_grad_ref(wrt)
        return grad_node


class Log(Operator):
    def __init__(self, input_node):
        super().__init__(input_node)

    def eval(self):
        node = self._input_nodes[0]
        if node.value <= 0:
            raise ValueError('Negtive value for log.')
        self.value = math.log(node.value)
        return self

    def eval_grad(self, wrt):
        grad_name = f'Grad({wrt.name})'
        grad_node = Constant(1 / wrt.value, grad_name)
        grad_node.set_grad_ref(wrt)
        return grad_node


class Reciprocal(Div):
    def __init__(self, input_node):
        one = Ones('numerator')
        super().__init__(one, input_node)
