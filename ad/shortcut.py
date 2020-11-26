from .ops import *      # noqa: F403


__all__ = ['constant', 'variable', 'add', 'neg',
           'minus', 'mul', 'div', 'pow', 'log',
           'ones', 'zeros']


def constant(value, name=None):
    return Constant(value, name)    # noqa: F405


def variable(value, name=None):
    return Variable(value, name)    # noqa: F405


def add(*nodes):
    return Add(*nodes)      # noqa: F405


def neg(node):
    return Neg(node)    # noqa: F405


def minus(*nodes):
    return Minus(*nodes)    # noqa: F405


def mul(*nodes):
    return Mul(*nodes)      # noqa: F405


def div(*nodes):
    return Div(*nodes)      # noqa: F405


def pow(*nodes):
    return Pow(*nodes)      # noqa: F405


def log(node):
    return Log(node)        # noqa: F405


def ones(node):
    return Ones(node)        # noqa: F405


def zeros(node):
    return Zeros(node)        # noqa: F405
