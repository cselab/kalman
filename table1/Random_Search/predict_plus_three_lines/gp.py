import collections
import copy
import io
import numpy as np
import random
import re
import struct
import subprocess
import sys
import tempfile
try:
    import graphlib
except ImportError:
    graphlib = None


class UnknownFormatString(Exception):
    pass


class CycleGraph(Exception):
    pass


class TooManyInputs(Exception):
    pass


class GraphlibNotFound(ModuleNotFoundError):

    def __init__(self):
        super().__init__(
            "The `graphlib` module is not available. "
            "It was introduced in Python 3.9. "
            f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
        )


dtype = np.dtype("uint8")
max_val = 256
graphviz_colors = "red", "blue", "green", "orange", "purple", "brown", "pink", "cyan", "magenta"


def execute(g, gen, x):
    rn = reachable_nodes(g, gen)
    Cost = 0
    values = {i: copy.copy(x[i]) for i in range(g.i)}
    for n in rn:
        arity = g.arity[gen[n, 0]]
        inputs = [values[i] for i in gen[n, 1:1 + arity]]
        params = gen[n, 1 + g.a:]
        values[n] = g.nodes[gen[n, 0]](inputs, params)
    return [values[j] for j in gen[g.i + g.n:, 1]]


def rand(g):
    gen = np.zeros((g.i + g.n + g.o, 1 + g.a + g.p), dtype=np.uint8)
    for j in range(g.n):
        gen[g.i + j, 0] = random.randrange(len(g.names))
        for k in range(g.a):
            gen[g.i + j, 1 + k] = random.randrange(g.i + j)
        for k in range(g.p):
            gen[g.i + j, 1 + g.a + k] = random.randrange(max_val)
    for j in range(g.o):
        gen[g.i + g.n + j, 1] = random.randrange(g.i + g.n)
    return gen


def as_image(g, gen, path, fmt="png", *args):
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as t:
        t.write(as_graphviz(g, gen))
        t.flush()
        subprocess.run(["dot", t.name, "-T", fmt, "-o", path, *args])


def as_string(g, gen, All=False):
    o = io.StringIO()
    rn = reachable_nodes(g, gen)
    for i in range(g.i):
        o.write("%3d: input\n" % i)
    for i in range(g.n):
        if All or g.i + i in rn:
            o.write("%3d: %s" % (g.i + i, g.names[gen[g.i + i, 0]]))
            p = io.StringIO()
            args = g.args[gen[g.i + i, 0]]
            for j in range(args):
                if j > 0:
                    p.write(", ")
                p.write("%d" % gen[g.i + i, 1 + g.a + j])
            o.write(" [%s]" % p.getvalue())
            for j in range(g.arity[gen[g.i + i, 0]]):
                o.write(" %d" % gen[g.i + i, 1 + j])
            if g.i + i in rn:
                o.write(" *")
            o.write("\n")
    for i in range(g.o):
        o.write("%3d: output %d\n" % (g.i + g.n + i, gen[g.i + g.n + i, 1]))
    return o.getvalue()


def as_graphviz(g, gen):
    o = io.StringIO()
    o.write("digraph {\n")
    for j in range(g.i):
        o.write(f"  {j} [label = i{j}]\n")
    for n in reachable_nodes(g, gen):
        arity = g.arity[gen[n, 0]]
        args = g.args[gen[n, 0]]
        o.write(f'  {n} [label = "{g.names[gen[n, 0]]}')
        for j in range(args):
            o.write(f", {gen[n, 1 + g.a + j]}")
        o.write('"]\n')
        for j in range(arity):
            if arity == 1:
                o.write(f"  {gen[n, 1 + j]} -> {n}\n")
            else:
                k = j % len(graphviz_colors)
                o.write(
                    f"  {gen[n, 1 + j]} -> {n} [color = {graphviz_colors[j]}]\n"
                )
    for j in range(g.o):
        o.write(f"  {g.i + g.n + j} [label = o{j}]\n")
        o.write(f"  {gen[g.i + g.n + j, 1]} -> {g.i + g.n + j}\n")
    o.write("}\n")
    return o.getvalue()


def build(g, verts, edgs, params):
    if graphlib == None:
        raise GraphlibNotFound
    assert max(v for edg in edgs for v in edg) < len(verts)
    assert min(v for edg in edgs for v in edg) >= 0
    graph = graphlib.TopologicalSorter()
    for x, y in edgs:
        graph.add(y, x)
    try:
        topo2input = tuple(graph.static_order())
    except graphlib.CycleError as exc:
        raise CycleGraph from exc
    input2topo = {x: i for i, x in enumerate(topo2input)}
    Names = {name: index for index, name in enumerate(g.names)}
    gen = np.zeros((g.i + g.n + g.o, 1 + g.a + g.p), dtype=np.uint8)
    cnt = 0
    D = {}
    for i, j in enumerate(topo2input):
        v = verts[j]
        if re.match("^o[0-9]+$", v):
            D[i] = g.i + g.n + int(v[1:])
        elif re.match("^i[0-9]+$", v):
            D[i] = int(v[1:])
        else:
            D[i] = g.i + cnt
            gen[D[i], 0] = Names[v]
            cnt += 1
    A = collections.defaultdict(int)
    for i, j in edgs:
        x = input2topo[i]
        y = input2topo[j]
        j0 = D[x]
        j1 = D[y]
        if A[y] >= g.a:
            raise TooManyInputs
        gen[j1, 1 + A[y]] = j0
        A[y] += 1
    for i, param in enumerate(params):
        x = input2topo[i]
        for k, p in enumerate(param):
            gen[D[x], 1 + g.a + k] = p
    return gen


def reachable_nodes(g, gen):
    q = {x for x in gen[g.i + g.n:, 1]}
    ans = set()
    while q:
        n = q.pop()
        if n >= g.i:
            ans.add(n)
            arity = g.arity[gen[n, 0]]
            adj = gen[n, 1:1 + arity]
            q.update(adj)
    return sorted(ans)


def serial(fmt, *args):
    assert len(fmt) == len(args)
    buf = bytearray()
    for f, a in zip(fmt, args):
        if f == "i":
            buf.extend(struct.pack("<i", a))
        elif f == "S":
            buf.extend(struct.pack("<i", len(a)))
            for s in a:
                buf.extend(struct.pack("<i", len(s)))
                buf.extend(s.encode("utf-8"))
        elif f == "I":
            buf.extend(struct.pack("<i", len(a)))
            for s in a:
                buf.extend(struct.pack("<i", s))
        elif f == "y":
            buf.extend(struct.pack("<i", len(a)))
            for s in a:
                assert s.dtype == dtype
                buf.extend(struct.pack("<i", s.ndim))
                buf.extend(struct.pack("<%ii" % s.ndim, *s.shape))
                buf.extend(s.tobytes())
        else:
            raise UnknownFormatString("Unknown format string: %s" % f)
    return buf


def deserial(fmt, buf):
    offset = 0
    ans = []
    for f in fmt:
        if f == "i":
            a, = struct.unpack_from("<i", buf, offset)
            offset += struct.calcsize("<i")
            ans.append(a)
        elif f == "S":
            n_strings, = struct.unpack_from("<i", buf, offset)
            offset += struct.calcsize("<i")
            strings = []
            for i in range(n_strings):
                size, = struct.unpack_from("<i", buf, offset)
                offset += struct.calcsize("<i")
                value, = struct.unpack_from("%ds" % size, buf, offset)
                offset += struct.calcsize("%ds" % size)
                strings.append(value.decode("utf-8"))
            ans.append(strings)
        elif f == "I":
            size, = struct.unpack_from("<i", buf, offset)
            offset += struct.calcsize("<i")
            value = struct.unpack_from("<%di" % size, buf, offset)
            offset += struct.calcsize("<%di" % size)
            ans.append(value)
        elif f == "y":
            n_arrays, = struct.unpack_from("<i", buf, offset)
            offset += struct.calcsize("<i")
            arrays = []
            for i in range(n_arrays):
                ndim, = struct.unpack_from("<i", buf, offset)
                offset += struct.calcsize("<i")
                shape = []
                for i in range(ndim):
                    d, = struct.unpack_from("<i", buf, offset)
                    offset += struct.calcsize("<i")
                    shape.append(d)
                array = np.ndarray(shape, dtype, buf, offset)
                arrays.append(array)
                offset += array.size
            ans.append(arrays)
        else:
            raise UnknownFormatString("Unknown format string: %s" % f)
    return ans
