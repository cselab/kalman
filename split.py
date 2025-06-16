import wavegp
import sys
import numpy as np
import random
import subprocess


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
y0 = 48, -16, 16, 16, 48, 0, 28, -24

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
gen0 = wavegp.rand(g)
gen1 = wavegp.rand(g)
gen2 = wavegp.build(
    g,
    #  0     1    2        3    4       5     6
    ["i0", "i1", "Minus", "U", "Plus", "o0", "o1"],
    [(1, 2), (0, 2), (2, 3), (0, 4), (3, 4), (4, 5), (2, 6)],
    [])
wavegp.as_image(g, gen2, "split.png")

for gen in gen0, gen1, gen2:
    sys.stdout.write(wavegp.as_string(g, gen))
    y = execute(gen, x0)
    sys.stdout.write("cost: %g\n\n" % diff(y, y0))
    with open("split.gv", "w") as f:
        f.write(wavegp.as_graphviz(g, gen))
