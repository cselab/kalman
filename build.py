import wavegp
import sys


class g:
    pass


g.names = "Backward_X", "Forward_X", "Backward_Y", "Forward_Y", "Plus", "Minus"
Names = {name: index for index, name in enumerate(g.names)}
g.arity = 1, 1, 1, 1, 2, 2
g.args = 1, 0, 1, 0, 0, 2
# input, maximum node, output, arity, parameters
g.i = 1
g.n = 10
g.o = 1
g.a = 2
g.p = 2
gen0 = wavegp.build(g, ["i0", "Backward_Y", "Backward_X", "Minus", "o0"],
                    [(0, 1), (0, 2), (1, 3), (2, 3),
                     (3, 4)], [[], [10], [20], [40, 30], []])
gen1 = wavegp.build(g, ["i0", "Backward_Y", "Backward_X", "Minus", "o0"],
                    [(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)], [])
fmt = "iiiiiSIIy"
buf = wavegp.serial(fmt, g.i, g.n, g.o, g.a, g.p, g.names, g.arity, g.args,
                    [gen0, gen1])
with open(sys.argv[1], "wb") as f:
    f.write(buf)
