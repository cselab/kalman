import wavegp
import sys


class g:
    pass


fmt = "iiiiiSIIy"
path = sys.argv[2]
with open(sys.argv[1], "rb") as f:
    buf = f.read()
    g.i, g.n, g.o, g.a, g.p, g.names, g.arity, g.args, genes = wavegp.deserial(
        fmt, buf)

with open(path, "w") as f:
    for gen in genes:
        f.write(wavegp.as_graphviz(g, gen))
sys.stderr.write("graphviz.py: %s\n" % path)
