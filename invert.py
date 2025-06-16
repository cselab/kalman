import wavegp
import sys
import numpy as np
import random
import subprocess
import copy
import graphlib


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
g.args = 0, 2, 0
# input, maximum node, output, arity, parameters
g.i = 2
g.n = 3
g.o = 2
g.a = 2
g.p = 2
gen0 = wavegp.build(
    g,
    #  0     1    2        3    4       5     6
    ["i0", "i1", "Minus", "U", "Plus", "o0", "o1"],
    [(1, 2), (0, 2), (2, 3), (0, 4), (3, 4), (4, 5), (2, 6)],
    [[], [], [10, 20], [], [], [], []])

to_inv = {"Minus": "Plus", "Plus": "Minus"}
assert (g.i == g.o)
gen = gen0
rn = wavegp.reachable_nodes(g, gen)
node2id = {x: i for i, x in enumerate(rn)}
edgs = []
verts = [f"i%d" % i for i in range(g.i)]
inv_verts = [f"o%d" % i for i in range(g.i)]
params = [[] for i in range(g.i)]
for j in range(g.i, g.i + g.n):
    if j in node2id:
        v = g.names[gen[j, 0]]
        verts.append(v)
        inv_verts.append(to_inv.get(v, v))
        arity = g.arity[gen[j, 0]]
        for k in gen[j, 1:1 + arity]:
            l = k if k < g.i else node2id[k] + g.i
            edgs.append((l, node2id[j] + g.i))
        args = g.args[gen[j, 0]]
        params.append([gen[j, 1 + g.a + l] for l in range(args)])
n = len(rn)
for i in range(g.o):
    j = gen[g.i + g.n + i, 1]
    j = j if j < g.i else node2id[j] + g.i
    k = g.i + n + i
    edgs.append((j, k))
verts.extend(f"o%d" % i for i in range(g.o))
inv_verts.extend(f"i%d" % i for i in range(g.o))
params.extend([] for i in range(g.o))


def dp(k, hist):
    if k == len(edgs):
        return [hist]
    i, j = edgs[k]
    if j + g.o >= len(verts):
        return dp(k + 1, hist + [(j, i)])
    else:
        return dp(k + 1, hist + [(j, i)]) + dp(k + 1, hist + [(i, j)])


print(verts, edgs)
for e in dp(0, []):
    try:
        gen = wavegp.build(g, inv_verts, e, params)
    except (wavegp.TooManyInputs, wavegp.CycleGraph):
        continue
    print(inv_verts, e)
    print(wavegp.as_string(g, gen))
