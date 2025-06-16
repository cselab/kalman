import wavegp
import sys


class g:
    pass


fmt = "iiiiiSIIy"
prefix = sys.argv[2]
with open(sys.argv[1], "rb") as f:
    g.i, g.n, g.o, g.a, g.p, g.names, g.arity, g.args, genes = wavegp.deserial(
        fmt, f.read())
for i, gen in enumerate(genes):
    path = "%s.%i.png" % (prefix, i)
    wavegp.as_image(g, gen, path)
    sys.stderr.write("png.py: %s\n" % path)
