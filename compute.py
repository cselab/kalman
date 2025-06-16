import wavegp
import sys


class g:
    pass


def evaluate(gen):
    topo = stopo(gen)
    Cost = 0
    values = {i: x[i] for i in range(g.i)}
    for n in topo:
        arity = g.nodes[gen[n, 0]].arity
        inputs = [values[i] for i in gen[n, 1:1 + arity]]
        params = gen[n, 1 + g.a:]
        values[n] = g.nodes[gen[n, 0]].call(inputs, params)
    return [values[j] for j in gen[g.i + g.n:, 1]]


g.names = "Backward_X", "Forward_X", "Backward_Y", "Forward_Y", "Plus", "Minus"
Names = {name: index for index, name in enumerate(g.names)}
g.arity = 1, 1, 1, 1, 2, 2
g.args = 0, 0, 0, 0, 0, 0
# input, maximum node, output, arity, parameters
g.i = 1
g.n = 10
g.o = 1
g.a = 2
g.p = 0
gen = wavegp.build(g, ["i0", "Backward_Y", "Backward_X", "Minus", "o0"],
                   [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], [])
sys.stdout.write(wavegp.as_string(g, gen))
