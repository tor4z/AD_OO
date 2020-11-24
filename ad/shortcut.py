from .ops import Constant, Variable


__all__ = ['constant', 'variable']


def constant(value, name=None):
    return Constant(value, name)


def variable(value, name=None):
    return Variable(value, name)
