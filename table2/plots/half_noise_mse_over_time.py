import numpy as np
import matplotlib.pyplot as plt
import wavegp  # Custom GP module

plt.rcParams.update({
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'legend.fontsize': 25,
    'figure.titlesize': 30
})


class g:
    pass


def minus(inp, args):
    return inp[0] - inp[1]


def matmul(inp, args):
    return inp[0] @ inp[1]


def add(inp, args):
    return inp[0] + inp[1]


def transpose(inp, args):
    return inp[0].T


def inv(inp, args):
    return np.linalg.inv(inp[0])


g.nodes = (matmul, minus, add, transpose, inv)
g.names = ("matmul", "minus", "add", "transpose", "inv")
g.arity = (2, 2, 2, 1, 1)
g.args = (0, 0, 0, 0, 0)
g.i = 6
g.n = 19
g.o = 6
g.a = 2
g.p = 0
g.lmb = 1000


def from_graphviz(g, dot_str):
    import re
    edges = []
    node_labels = {}
    for line in dot_str.splitlines():
        line = line.strip()
        if not line or line.startswith("digraph") or line in {"{", "}"}:
            continue
        label_match = re.match(r'^(\d+)\s+\[label\s*=\s*"?([^"+]+?)"?\]', line)
        edge_match = re.match(r'^(\d+)\s*->\s*(\d+)', line)
        if label_match:
            nid = int(label_match.group(1))
            label = label_match.group(2)
            node_labels[nid] = label
        elif edge_match:
            src = int(edge_match.group(1))
            tgt = int(edge_match.group(2))
            edges.append((src, tgt))
    input_map = {}
    for src, tgt in edges:
        input_map.setdefault(tgt, []).append(src)
    max_node_id = max(node_labels.keys())
    total_rows = max(max_node_id + 1, g.i + g.n + g.o)
    gen = np.zeros((total_rows, 1 + g.a + g.p), dtype=int)
    node_id_to_row = {}
    for nid, label in node_labels.items():
        if label.startswith("i") and not label.startswith("inv"):
            idx = int(label[1:])
            node_id_to_row[nid] = idx
    for nid in range(total_rows):
        label = node_labels.get(nid)
        if label in g.names:
            op_idx = g.names.index(label)
            gen[nid, 0] = op_idx
            node_id_to_row[nid] = nid
            inputs = input_map.get(nid, [])
            for j in range(min(g.a, len(inputs))):
                gen[nid, 1 + j] = node_id_to_row.get(inputs[j], inputs[j])
        elif label is None:
            continue
        elif not label.startswith("i") and not label.startswith(
                "o") and not label.startswith("inv"):
            gen[nid, 0] = g.names.index("add")
            gen[nid, 1:] = 0
            node_labels[nid] = "add"
    for o in range(g.o):
        for nid, label in node_labels.items():
            if label == f"o{o}":
                srcs = input_map.get(nid, [])
                if srcs:
                    gen[nid, 0] = 0
                    gen[nid, 1] = node_id_to_row.get(srcs[0], srcs[0])
    return gen


best_graph_dot = '''digraph {
  0 [label = i0]
  1 [label = i1]
  2 [label = i2]
  3 [label = i3]
  4 [label = i4]
  5 [label = i5]
  6 [label = "transpose"]
  4 -> 6
  7 [label = "add"]
  6 -> 7 [color = red]
  0 -> 7 [color = blue]
  8 [label = "minus"]
  2 -> 8 [color = red]
  7 -> 8 [color = blue]
  9 [label = "add"]
  8 -> 9 [color = red]
  1 -> 9 [color = blue]
  10 [label = "transpose"]
  9 -> 10
  12 [label = "matmul"]
  10 -> 12 [color = red]
  2 -> 12 [color = blue]
  13 [label = "transpose"]
  12 -> 13
  14 [label = "add"]
  13 -> 14 [color = red]
  7 -> 14 [color = blue]
  15 [label = "inv"]
  14 -> 15
  17 [label = "add"]
  7 -> 17 [color = red]
  6 -> 17 [color = blue]
  19 [label = "minus"]
  17 -> 19 [color = red]
  5 -> 19 [color = blue]
  20 [label = "matmul"]
  17 -> 20 [color = red]
  15 -> 20 [color = blue]
  21 [label = "minus"]
  4 -> 21 [color = red]
  20 -> 21 [color = blue]
  22 [label = "inv"]
  8 -> 22
  24 [label = "transpose"]
  12 -> 24
  25 [label = o0]
  0 -> 25
  26 [label = o1]
  22 -> 26
  27 [label = o2]
  24 -> 27
  28 [label = o3]
  3 -> 28
  29 [label = o4]
  19 -> 29
  30 [label = o5]
  21 -> 30
}
'''

random_search = '''digraph {
  0 [label = i0]
  1 [label = i1]
  2 [label = i2]
  3 [label = i3]
  4 [label = i4]
  5 [label = i5]
  6 [label = "inv"]
  1 -> 6
  7 [label = "matmul"]
  6 -> 7 [color = red]
  6 -> 7 [color = blue]
  8 [label = "add"]
  7 -> 8 [color = red]
  7 -> 8 [color = blue]
  9 [label = "add"]
  7 -> 9 [color = red]
  3 -> 9 [color = blue]
  10 [label = "matmul"]
  8 -> 10 [color = red]
  7 -> 10 [color = blue]
  11 [label = "add"]
  10 -> 11 [color = red]
  4 -> 11 [color = blue]
  13 [label = "inv"]
  2 -> 13
  15 [label = "inv"]
  11 -> 15
  17 [label = "matmul"]
  4 -> 17 [color = red]
  13 -> 17 [color = blue]
  18 [label = "add"]
  11 -> 18 [color = red]
  9 -> 18 [color = blue]
  19 [label = "add"]
  17 -> 19 [color = red]
  11 -> 19 [color = blue]
  23 [label = "matmul"]
  15 -> 23 [color = red]
  18 -> 23 [color = blue]
  25 [label = o0]
  4 -> 25
  26 [label = o1]
  23 -> 26
  27 [label = o2]
  3 -> 27
  28 [label = o3]
  19 -> 28
  29 [label = o4]
  11 -> 29
  30 [label = o5]
  17 -> 30
}
'''

best_graph = from_graphviz(g, best_graph_dot)
random_graph = from_graphviz(g, random_search)


def evaluate_best_graph(x, F, P, Q, z, R):
    try:
        _, P_new, _, _, _, x_new = wavegp.execute(
            g, best_graph,
            [x.copy(),
             F.copy(),
             P.copy(),
             Q.copy(),
             z.copy(),
             R.copy()])
    except Exception as e:
        print("Graph execution error:", e)
        x_new = np.full_like(x, np.nan)
        P_new = P
    return x_new, P_new


def evaluate_random_graph(x, F, P, Q, z, R):
    try:
        _, P_new, _, _, _, x_new = wavegp.execute(
            g, random_graph,
            [x.copy(),
             F.copy(),
             P.copy(),
             Q.copy(),
             z.copy(),
             R.copy()])
    except Exception as e:
        print("Random graph execution error:", e)
        x_new = np.full_like(x, np.nan)
        P_new = P
    return x_new, P_new


# === Settings ===
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
n_runs = 1000
time_steps = 500
time = np.arange(time_steps + 1)


def our_approximation(x, F, P, Q, z, R):
    xp = F @ x
    P_new = F @ P @ F.T + Q
    y = z - xp
    S = P_new + R + F.min(axis=1)[:, None] * 1.2
    inv_S = np.linalg.inv(S)
    K = (P_new @ inv_S) * 0.85
    x = xp + K @ y
    P = (P_new - K @ S @ K.T) * 0.95
    return xp, P, y, S, K, x


def graph_approximate(x, F, P, Q, z, R):
    A = R.T + x
    y = P - A
    S = M = (y + F).T @ P + A
    K = np.linalg.inv(S)
    xp = R - (A + R.T) @ K
    P = np.linalg.inv(y)
    return xp, P, y, S, K, x

    return xp, P, y, S, K, x


class KalmanFilter:

    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x = x.copy()

    def predict(self, u=np.zeros(2)):
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - (self.H @ self.x)
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x


def simulate_run(seed):
    rng = np.random.default_rng(seed)
    x = np.array([0, 0], dtype=float)
    P = np.eye(dim)
    x_approx = x.copy()
    x_graph = x.copy()
    x_random = x.copy()
    P_graph = P.copy()
    P_random = P.copy()
    kf = KalmanFilter(F, B, H, Q, R, P.copy(), x.copy())

    true_states = [x.copy()]
    observations = [x.copy()]
    approx_states = [x.copy()]
    graph_states = [x.copy()]
    random_states = [x.copy()]
    kf_states = [x.copy()]

    for _ in range(time_steps):
        x = F @ x + cQ @ abs(rng.normal(0, 1, dim))
        z = H @ x + cR @ abs(rng.normal(0, 1, dim))

        _, P, _, _, _, x_approx = our_approximation(x_approx.copy(), F, P, Q,
                                                    z, R)
        x_graph, P_graph = evaluate_best_graph(x_graph.copy(), F,
                                               P_graph.copy(), Q, z, R)
        x_random, P_random = evaluate_random_graph(x_random.copy(), F,
                                                   P_random.copy(), Q, z, R)
        kf.predict()
        kf_state = kf.update(z)

        true_states.append(x.copy())
        observations.append(z.copy())
        approx_states.append(x_approx.copy())
        graph_states.append(x_graph.copy())
        random_states.append(x_random.copy())
        kf_states.append(kf_state.copy())

    return (np.array(true_states), np.array(observations),
            np.array(approx_states), np.array(kf_states),
            np.array(graph_states), np.array(random_states))


true_stack = []
obs_stack = []
approx_stack = []
kf_stack = []
graph_stack = []
random_stack = []

for i in range(n_runs):
    t, o, a, k, g_out, r_out = simulate_run(seed=42 + i)
    true_stack.append(t)
    obs_stack.append(o)
    approx_stack.append(a)
    kf_stack.append(k)
    graph_stack.append(g_out)
    random_stack.append(r_out)

true_stack = np.stack(true_stack)
obs_stack = np.stack(obs_stack)
approx_stack = np.stack(approx_stack)
kf_stack = np.stack(kf_stack)
graph_stack = np.stack(graph_stack)
random_stack = np.stack(random_stack)


def compute_mse_and_se(estimates, truths):
    errors = np.linalg.norm(estimates - truths, axis=2)**2
    mse = np.mean(errors, axis=0)
    se = np.std(errors, axis=0) / np.sqrt(errors.shape[0])
    return mse, se


mse_obs, se_obs = compute_mse_and_se(obs_stack, true_stack)
mse_approx, se_approx = compute_mse_and_se(approx_stack, true_stack)
mse_kf, se_kf = compute_mse_and_se(kf_stack, true_stack)
mse_graph, se_graph = compute_mse_and_se(graph_stack, true_stack)
mse_random, se_random = compute_mse_and_se(random_stack, true_stack)

plt.figure(figsize=(12, 8))
plt.plot(time, mse_obs, '--', label='Observations')
plt.fill_between(time, mse_obs - se_obs, mse_obs + se_obs, alpha=0.2)
plt.plot(time, mse_kf, ':', label='Kalman Filter')
plt.fill_between(time, mse_kf - se_kf, mse_kf + se_kf, alpha=0.2)
plt.plot(time, mse_random, '--', label='Random Search')
plt.fill_between(time,
                 mse_random - se_random,
                 mse_random + se_random,
                 alpha=0.2)
plt.plot(time, mse_approx, '-.', label='Funsearch')
plt.fill_between(time,
                 mse_approx - se_approx,
                 mse_approx + se_approx,
                 alpha=0.2)
plt.plot(time, mse_graph, '-', label='CGP')
plt.fill_between(time, mse_graph - se_graph, mse_graph + se_graph, alpha=0.2)
plt.xlabel("Time Step")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("MSE_over_time_half_noise.png", dpi=1200)
plt.savefig("MSE_over_time_half_noise.pdf", dpi=1200)
plt.show()
