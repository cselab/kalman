import wavegp
import sys
import numpy as np
import random


class g:
    pass


def Forward_X(inp, args):
    x, = inp
    y = np.empty((N, N))
    for j in range(N):
        for i in range(N - 1):
            y[i, j] = x[i + 1, j] - x[i, j]
        y[N - 1, j] = x[0, j] - x[N - 1, j]
    return y


def Backward_X(inp, args):
    x, = inp
    y = np.empty((N, N))
    for j in range(N):
        for i in range(1, N):
            y[i, j] = x[i, j] - x[i - 1, j]
        y[0, j] = x[0, j] - x[N - 1, j]
    return y


def Forward_Y(inp, args):
    x, = inp
    y = np.empty((N, N))
    for i in range(N):
        for j in range(N - 1):
            y[i, j] = x[i, j + 1] - x[i, j]
        y[i, N - 1] = x[i, 0] - x[i, N - 1]
    return y


def Backward_Y(inp, args):
    x, = inp
    y = np.empty((N, N))
    for i in range(N):
        for j in range(1, N):
            y[i, j] = x[i, j] - x[i, j - 1]
        y[i, 0] = x[i, 0] - x[i, N - 1]
    return y


def Plus(inp, args):
    x, y = inp
    return x + y


def Minus(inp, args):
    x, y = inp
    return x - y


def diff(a, b):
    diff = np.subtract(a[1:-1, 1:-1], b[1:-1, 1:-1])
    return np.mean(diff**2)


random.seed(3)
N = 10
Y, X = np.meshgrid(range(N), range(N))
x0 = Y**2 + 2 * X**2
y0 = np.full((N, N), 2 + 2 * 2)

g.nodes = Backward_X, Forward_X, Backward_Y, Forward_Y, Plus, Minus
g.names = "Backward_X", "Forward_X", "Backward_Y", "Forward_Y", "Plus", "Minus"
g.arity = 1, 1, 1, 1, 2, 2
g.args = 0, 0, 0, 0, 0, 0
# input, maximum node, output, arity, parameters
g.i = 1
g.n = 10
g.o = 1
g.a = 2
g.p = 0
gen0 = wavegp.build(g, ["i0", "Backward_Y", "Backward_X", "Minus", "o0"],
                    [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], [])
gen1 = wavegp.rand(g)
gen2 = wavegp.build(
    g,
    ["i0", "Backward_X", "Forward_X", "Backward_Y", "Forward_Y", "Plus", "o0"],
    [(0, 1), (1, 2), (2, 5), (0, 3), (3, 4), (4, 5), (5, 6)], [])

gen3 = wavegp.build(
    g,
    ["i0", "Backward_Y", "Forward_Y", "Backward_X", "Forward_X", "Plus", "o0"],
    [(0, 1), (1, 2), (2, 5), (0, 3), (3, 4), (4, 5), (5, 6)], [])

for gen in gen0, gen1, gen2, gen3:
    sys.stdout.write(wavegp.as_string(g, gen, All=True))
    y, = wavegp.execute(g, gen, [x0])
    sys.stdout.write("cost: %g\n\n" % diff(y, y0))
