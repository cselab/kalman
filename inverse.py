import wavegp
import sys
import numpy as np
import random
import subprocess


class g:
    pass


def diff(a, b):
    diff = np.subtract(a, b)
    return np.mean(diff**2)


def Even(inp, args):
    x, = inp
    y = np.zeros(N)
    for i in range(N // 2):
        y[i] = x[2 * i]
    return y


def Odd(inp, args):
    x, = inp
    y = np.zeros(N)
    for i in range(N // 2):
        y[i] = x[2 * i + 1]
    return y


def Plus(inp, args):
    x, y = inp
    z = np.zeros(N)
    for i in range(N):
        z[i] = x[i] + y[i]
    return z


def Minus(inp, args):
    x, y = inp
    z = np.zeros(N)
    for i in range(N):
        z[i] = x[i] - y[i]
    return z


def P(inp, args):
    x, = inp
    return x


def U(inp, args):
    x, = inp
    return x / 2


def Merge(inp, args):
    x, y = inp
    z = np.empty(N)
    for i in range(N // 2):
        z[2 * i] = x[i]
        z[2 * i + 1] = y[i]
    return z


random.seed(2)
N = 8
Y, X = np.meshgrid(range(N), range(N))
x0 = 56, 40, 8, 24, 48, 48, 40, 16
y0 = 48, -16, 16, 16, 48, 0, 28, -24

g.nodes = Even, Odd, Plus, Minus, P, U, Merge
g.names = "Even", "Odd", "Plus", "Minus", "P", "U", "Merge"
g.arity = 1, 1, 2, 2, 1, 1, 2
g.args = 0, 0, 0, 0, 0, 0, 0
# input, maximum node, output, arity, parameters
g.i = 1
g.n = 6
g.o = 1
g.a = 2
g.p = 0
gen_forward = wavegp.build(
    g,
    #  0      1       2        3    4    5        6     7
    ["i0", "Odd", "Even", "Minus", "U", "Plus", "Merge", "o0"],
    [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 5), (4, 5), (5, 6), (3, 6),
     (6, 7)],
    [])

gen_backward = wavegp.build(
    g,
    #  0      1       2    3        4       5        6     7
    ["i0", "Odd", "Even", "U", "Minus", "Plus", "Merge", "o0"],
    [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (1, 5), (4, 5), (4, 6),
     (5, 6), (6, 7)],
    [])

for gen in [gen_backward]:
    sys.stdout.write(wavegp.as_string(g, gen))
    x, = wavegp.execute(g, gen, [y0])
    sys.stdout.write("cost: %g\n\n" % diff(x, x0))
    with open("inverse.gv", "w") as f:
        f.write(wavegp.as_graphviz(g, gen))
    subprocess.run(["dot", "inverse.gv", "-T", "svg", "-o", "img/inverse.svg"])
