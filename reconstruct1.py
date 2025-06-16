import wavegp
import sys
import numpy as np
import random
import subprocess
import statistics
import collections


class g:
    pass


def rand(project):
    gen = wavegp.rand(g)
    project(gen)
    return gen


def project0(gen):
    for j in range(g.n):
        if gen[g.i + j, 0] == Names["Odd"]:
            gen[g.i + j, 1] = 0
        elif gen[g.i + j, 0] == Names["Even"]:
            gen[g.i + j, 1] = 0


def project_forward(gen):
    gen[:] = forward0[:]
    return
    j = gen[g.i + g.n + 0, 1]
    gen[j, 0] = Names["Merge"]
    gen[gen[j, 1 + 0], 0] = Names["Plus"]
    gen[gen[j, 1 + 1], 0] = Names["Minus"]
    project0(gen)


def project_backward(gen):
    j = gen[g.i + g.n + 0, 1]
    gen[j, 0] = Names["Merge"]
    gen[gen[j, 1 + 0], 0] = Names["Minus"]
    gen[gen[j, 1 + 1], 0] = Names["Plus"]
    project0(gen)


def fun(forward, backward):
    cost = []
    for x0 in xx:
        y, = wavegp.execute(g, forward, [x0])
        i, j, *rest = np.argsort(y[1::2])
        y[2 * i + 1] = 0
        y[2 * j + 1] = 0
        x, = wavegp.execute(g, backward, [y])
        l = diff(x, x0)
        cost.append(l)
    ans = statistics.mean(cost)
    return ans


def example():
    p = 2
    q = 10
    x = [random.randint(-p, p)]
    for i in range(N - 1):
        x.append(x[-1] + random.randint(-p, p))
        p, q = q, p
    return np.array(x, dtype=float)


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
N = 1 << 3
g.nodes = Even, Odd, Plus, Minus, U, Merge
g.names = "Even", "Odd", "Plus", "Minus", "U", "Merge"
Names = {name: i for i, name in enumerate(g.names)}
g.arity = 1, 1, 2, 2, 1, 2
g.args = 0, 0, 0, 0, 0, 0
# input, maximum node, output, arity, parameters
g.i = 1
g.n = 6
g.o = 1
g.a = 2
g.p = 0
forward0 = wavegp.build(
    g,
    #  0      1       2        3    4    5        6     7
    ["i0", "Odd", "Even", "Minus", "U", "Plus", "Merge", "o0"],
    [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 5), (4, 5), (5, 6), (3, 6),
     (6, 7)],
    [])

backward0 = wavegp.build(
    g,
    #  0      1       2    3        4       5        6     7
    ["i0", "Odd", "Even", "U", "Minus", "Plus", "Merge", "o0"],
    [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (1, 5), (4, 5), (4, 6),
     (5, 6), (6, 7)],
    [])

xx = [example() for i in range(5)]
sys.stdout.write("cost: %g\n" % fun(forward0, backward0))
project_forward(forward0)
project_backward(backward0)
cost0 = fun(forward0, backward0)
best = sys.float_info.max, None, None

generation = 0
while True:
    forward = rand(project_forward)
    backward = rand(project_backward)
    cost = fun(forward, backward)
    if cost < best[0]:
        wavegp.as_image(g, forward, "best.forward.png")
        wavegp.as_image(g, backward, "best.backward.png")
        sys.stdout.write("forward\n" + wavegp.as_string(g, forward))
        sys.stdout.write("backward\n" + wavegp.as_string(g, backward))
        sys.stdout.write("\n")
        best = cost, forward, backward
    if generation % 10000 == 0:
        sys.stdout.write(f"{generation:015} {best[0]:.16e} {cost0:.16e}\n")
    generation += 1
