import wavegp
import sys
import numpy as np
import random
import subprocess
import pathlib


class g:
    pass


def execute(gen, x):
    xe = x[0::2]
    xo = x[1::2]
    ye, yo = wavegp.execute(g, gen, [xe, xo])
    y = np.empty(N, dtype=dtype)
    y[0::2] = ye
    y[1::2] = yo
    return y


def diff(a, b):
    diff = np.subtract(a, b, dtype=dtype)
    return np.mean(diff**2)


def Plus(inp, args):
    x, y = inp
    return np.add(x, y, dtype=dtype)


def Minus(inp, args):
    x, y = inp
    return np.subtract(x, y)


def P(inp, args):
    x, = inp
    return x


def U(inp, args):
    x, = inp
    return np.divide(x, 2, dtype=dtype)


dtype = float
random.seed(2)
N = 8
x0 = 56, 40, 8, 24, 48, 48, 40, 16
g.nodes = Plus, Minus, U
g.names = "Plus", "Minus", "U"
g.arity = 2, 2, 1
g.args = 0, 0, 0
# input, maximum node, output, arity, parameters
g.i = 2
g.n = 3
g.o = 2
g.a = 2
g.p = 0
gen_forward = wavegp.build(
    g,
    #  0     1    2        3    4       5     6
    ["i0", "i1", "Minus", "U", "Plus", "o0", "o1"],
    [(1, 2), (0, 2), (2, 3), (0, 4), (3, 4), (4, 5), (2, 6)],
    [])

gen_backward = wavegp.build(
    g,
    #  0     1    2        3    4       5     6
    ["i0", "i1", "U", "Minus", "Plus", "o0", "o1"],
    [
        (1, 2),  # U
        (0, 3),  # Minus
        (2, 3),
        (1, 4),  # Plus
        (3, 4),
        (3, 5),  # o0
        (4, 6),  # o1
    ],
    [])

y = execute(gen_forward, x0)
x = execute(gen_backward, y)
sys.stdout.write("cost1: %g\n" % diff(x, x0))
pathlib.Path("forward.gv").write_text(wavegp.as_graphviz(g, gen_forward))
pathlib.Path("backward.gv").write_text(wavegp.as_graphviz(g, gen_backward))
