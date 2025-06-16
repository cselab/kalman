import wavegp
import sys
import numpy as np
import random
import subprocess
import statistics
import pathlib
import os


class g:
    pass


def seen(a, b):
    a = a.tobytes()
    b = b.tobytes()
    ans = (a, b) in Hash
    if not ans:
        Hash.add((a, b))
    return ans


def compression_loss(forward, backward, x0):
    y = execute(forward, x0)
    i, j, *rest = np.argsort(np.abs(y[1::2]))
    y[2 * i + 1] = 0
    y[2 * j + 1] = 0
    x = execute(backward, y)
    return diff(x, x0)


def mean_loss_forward(gen, x):
    y = execute(gen, x)
    m1 = np.mean(x)
    m2 = np.mean(y[::2])
    return (m1 - m2)**2


def mean_loss_backward(gen, x):
    y = execute(gen, x)
    m1 = np.mean(x[::2])
    m2 = np.mean(y)
    return (m1 - m2)**2


def fun(forward, backward):
    l1 = statistics.mean(compression_loss(forward, backward, x) for x in xx)
    l2 = statistics.mean(mean_loss_forward(forward, x) for x in xx)
    l3 = statistics.mean(mean_loss_backward(backward, x) for x in xx)
    return l1 + l2 + l3


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


def example(rnd):
    p = 1
    x = [rnd.randint(-p, p)]
    for i in range(N - 1):
        x.append(x[-1] + rnd.randint(-p, p))
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
seed = 123456 if len(sys.argv) < 2 else int(sys.argv[1])
sys.stdout.write(f"{os.getpid()}: seed: {seed}\n")
random.seed(seed)
N = 32
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

rnd = random.Random(1234)
xx = [example(rnd) for i in range(5)]
cost0 = fun(forward0, backward0)
best = sys.float_info.max, None, None
generation = 0
max_generation = 99999999999
with open("cost", "w") as cost_file:
    while True:
        while True:
            forward = wavegp.rand(g)
            backward = wavegp.rand(g)
            if not seen(forward, backward):
                break
        cost = fun(forward, backward)
        if cost <= best[0]:
            pathlib.Path("%010d.forward.gv" % generation).write_text(
                wavegp.as_graphviz(g, forward))
            pathlib.Path("%010d.backward.gv" % generation).write_text(
                wavegp.as_graphviz(g, backward))
            best = cost, forward, backward
            cost_file.write(f"{generation:010d} {best[0]:.16e} {cost0:.16e}\n")
            cost_file.flush()
        generation += 1
        if generation % 10000 == 1 or generation == max_generation:
            sys.stdout.write(
                f"{os.getpid()} {generation:010d} {len(Hash):09} {best[0]:.16e} {cost0:.16e}\n"
            )
        if generation == max_generation:
            break
