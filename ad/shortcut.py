from .ops import Constant, Variable, Add


__all__ = ['constant', 'variable', 'add']


def constant(value, name=None):
    return Constant(value, name)


def variable(value, name=None):
    return Variable(value, name)


def add(*nodes):
    return Add(*nodes)
