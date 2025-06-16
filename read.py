import wavegp
import sys


class g:
    pass


fmt = "iiiiiSIIy"
with open(sys.argv[1], "rb") as f:
    buf = f.read()
    g.i, g.n, g.o, g.a, g.p, g.names, g.arity, g.args, genes = wavegp.deserial(
        fmt, buf)
print("n_genes = ", len(genes))
for gen in genes:
    print(wavegp.as_string(g, gen, All=False))
