import wavegp
import sys
import numpy as np
import random
import subprocess
import statistics
import pathlib


class g:
    pass


def seen(a, b):
    a = a.tobytes()
    b = b.tobytes()
    ans = (a, b) in Hash
    if not ans:
        Hash.add((a, b))
    return ans


def fun(forward, backward):
    cost = []
    for x0 in xx:
        y = execute(forward, x0)
        i, j, *rest = np.argsort(y[1::2])
        y[2 * i + 1] = 0
        y[2 * j + 1] = 0
        x = execute(backward, y)
        l = diff(x, x0)
        cost.append(l)
    ans = statistics.mean(cost)
    return ans


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


def example():
    p = 2
    q = 10
    x = [random.randint(-p, p)]
    for i in range(N - 1):
        x.append(x[-1] + random.randint(-p, p))
        p, q = q, p
    return np.array(x, dtype=dtype)


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


Hash = set()
dtype = float
random.seed(123456)
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
forward0 = wavegp.build(
    g,
    #  0     1    2        3    4       5     6
    ["i0", "i1", "Minus", "U", "Plus", "o0", "o1"],
    [
        (1, 2),  # Minus
        (0, 2),
        (2, 3),  # U
        (0, 4),  # Plus
        (3, 4),
        (4, 5),  # o0
        (2, 6)
    ],  # o1
    [])

backward0 = wavegp.build(
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

xx = [example() for i in range(10)]
cost0 = fun(forward0, backward0)
best = sys.float_info.max, None, None
generation = 0
max_generation = 120000
while True:
    while True:
        forward = wavegp.rand(g)
        backward = wavegp.rand(g)
        if not seen(forward, backward):
            break
    cost = fun(forward, backward)
    if cost < best[0]:
        pathlib.Path("split3.forward.gv").write_text(
            wavegp.as_graphviz(g, forward))
        pathlib.Path("split3.backward.gv").write_text(
            wavegp.as_graphviz(g, backward))
        sys.stdout.write("forward\n" + wavegp.as_string(g, forward, All=True))
        sys.stdout.write("backward\n" +
                         wavegp.as_string(g, backward, All=True))
        sys.stdout.write("\n")
        best = cost, forward, backward
    generation += 1
    if generation % 10000 == 1 or generation == max_generation:
        sys.stdout.write(
            f"{generation:09} {len(Hash):09} {best[0]:.16e} {cost0:.16e}\n")
    if generation == max_generation:
        break
