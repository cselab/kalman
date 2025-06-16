from random import randrange
import numpy as np
import random
import wavegp
import sys


class g:
    pass


random.seed(2)
g.names = "Backward_X", "Forward_X", "Backward_Y", "Forward_Y", "Plus", "Minus"
g.arity = 1, 1, 1, 1, 2, 2
g.args = 0, 0, 0, 0, 0, 0
# input, maximum node, output, arity, parameters
g.i = 1
g.n = 10
g.o = 1
g.a = 2
g.p = 2
g.lmb = 8
genes = [wavegp.rand(g) for i in range(g.lmb)]
fmt = "iiiiiSIIy"
buf = wavegp.serial(fmt, g.i, g.n, g.o, g.a, g.p, g.names, g.arity, g.args,
                    genes)
with open(sys.argv[1], "wb") as f:
    f.write(buf)
