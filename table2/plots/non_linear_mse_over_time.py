best_graph_dot = '''digraph {
  0 [label = i0]
  1 [label = i1]
  2 [label = i2]
  3 [label = i3]
  4 [label = i4]
  5 [label = i5]
  6 [label = "transpose"]
  1 -> 6
  7 [label = "matmul"]
  1 -> 7 [color = red]
  0 -> 7 [color = blue]
  8 [label = "transpose"]
  6 -> 8
  9 [label = "minus"]
  4 -> 9 [color = red]
  7 -> 9 [color = blue]
  10 [label = "add"]
  8 -> 10 [color = red]
  6 -> 10 [color = blue]
  11 [label = "add"]
  10 -> 11 [color = red]
  2 -> 11 [color = blue]
  12 [label = "add"]
  11 -> 12 [color = red]
  5 -> 12 [color = blue]
  13 [label = "inv"]
  12 -> 13
  14 [label = "matmul"]
  10 -> 14 [color = red]
  13 -> 14 [color = blue]
  15 [label = "matmul"]
  14 -> 15 [color = red]
  9 -> 15 [color = blue]
  17 [label = "add"]
  0 -> 17 [color = red]
  15 -> 17 [color = blue]
  25 [label = o0]
  0 -> 25
  26 [label = o1]
  5 -> 26
  27 [label = o2]
  9 -> 27
  28 [label = o3]
  12 -> 28
  29 [label = o4]
  14 -> 29
  30 [label = o5]
  17 -> 30
}'''

random_search = '''digraph {
  0 [label = i0]
  1 [label = i1]
  2 [label = i2]
  3 [label = i3]
  4 [label = i4]
  5 [label = i5]
  6 [label = "matmul"]
  4 -> 6 [color = red]
  1 -> 6 [color = blue]
  8 [label = "add"]
  6 -> 8 [color = red]
  0 -> 8 [color = blue]
  9 [label = "add"]
  2 -> 9 [color = red]
  1 -> 9 [color = blue]
  12 [label = "minus"]
  6 -> 12 [color = red]
  0 -> 12 [color = blue]
  13 [label = "inv"]
  2 -> 13
  14 [label = "inv"]
  9 -> 14
  16 [label = "matmul"]
  8 -> 16 [color = red]
  14 -> 16 [color = blue]
  24 [label = "transpose"]
  14 -> 24
  25 [label = o0]
  12 -> 25
  26 [label = o1]
  2 -> 26
  27 [label = o2]
  5 -> 27
  28 [label = o3]
  24 -> 28
  29 [label = o4]
  13 -> 29
  30 [label = o5]
  16 -> 30
}'''

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
        print("Random graph error:", e)
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


def function_approximate(x, F, P, Q, z, R):
    x = np.array([
        0.04 * x[0]**3 - 1.8 * x[0] + 0.34 * np.sin(x[1]),
        0.14 * np.tanh(0.05 * x[0] * (x[1] + 0.8))
    ])
    xp = F.dot(x)
    scale_Q = (x[0] * x[1] + x[0]**2 + x[1]**2 + 0.6) * \
                (1 + 0.9 * (x[0] * x[1] + x[0]**2 + x[1]**2))
    P = F.dot(P.dot(F.T)) + Q * scale_Q
    y = z - xp
    scale_R = (x[0]**2 + x[1]**2 + 0.5) * (1 + 0.8 * (x[0]**2 + x[1]**2))
    S = P + R * scale_R
    inv_S = np.linalg.inv(S + 0.0002 * np.eye(S.shape[0]))
    K = P.dot(inv_S) * (0.85 * np.tanh(np.linalg.norm(y)))
    x = xp + K.dot(y)
    P = (np.eye(F.shape[0]) - K) * P * (0.5 + 0.05 * np.mean(y**2))
    return xp, P, y, S, K, x


def graph_approximate(x, F, P, Q, z, R):
    Ft = F.T
    y = z - (F @ x)
    S = Ft.T + Ft + P + Q
    K = Ft @ np.linalg.inv(S)
    P = K @ y
    xp = x + P
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
        self.x = np.array(
            [0.05 * self.x[0]**3 - 2 * self.x[0], 0.1 * np.sin(self.x[1])])
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
    x_rand = x.copy()
    P_graph = P.copy()
    P_rand = P.copy()
    kf = KalmanFilter(F, B, H, Q, R, P.copy(), x.copy())

    true_states = [x.copy()]
    observations = [x.copy()]
    approx_states = [x.copy()]
    graph_states = [x.copy()]
    rand_states = [x.copy()]
    kf_states = [x.copy()]

    for _ in range(time_steps):
        x = np.array([0.05 * x[0]**3 - 2 * x[0], 0.1 * np.sin(x[1])])
        x = F @ x + cQ @ rng.normal(0, 1, dim)
        z = H @ x + cR @ rng.normal(0, 1, dim)

        _, P, _, _, _, x_approx = function_approximate(x_approx.copy(), F, P,
                                                       Q, z, R)

        x_graph = np.array(
            [0.05 * x_graph[0]**3 - 2 * x_graph[0], 0.1 * np.sin(x_graph[1])])
        x_graph, P_graph = evaluate_best_graph(x_graph.copy(), F,
                                               P_graph.copy(), Q, z, R)

        x_rand = np.array(
            [0.05 * x_rand[0]**3 - 2 * x_rand[0], 0.1 * np.sin(x_rand[1])])
        x_rand, P_rand = evaluate_random_graph(x_rand.copy(), F, P_rand.copy(),
                                               Q, z, R)

        kf.predict()
        kf_state = kf.update(z)

        true_states.append(x.copy())
        observations.append(z.copy())
        approx_states.append(x_approx.copy())
        graph_states.append(x_graph.copy())
        rand_states.append(x_rand.copy())
        kf_states.append(kf_state.copy())

    return (np.array(true_states), np.array(observations),
            np.array(approx_states), np.array(kf_states),
            np.array(graph_states), np.array(rand_states))

true_stack, obs_stack, approx_stack, kf_stack, graph_stack, rand_stack = [], [], [], [], [], []
for i in range(n_runs):
    t, o, a, k, g_out, r_out = simulate_run(seed=42 + i)
    true_stack.append(t)
    obs_stack.append(o)
    approx_stack.append(a)
    kf_stack.append(k)
    graph_stack.append(g_out)
    rand_stack.append(r_out)

true_stack = np.stack(true_stack)
obs_stack = np.stack(obs_stack)
approx_stack = np.stack(approx_stack)
kf_stack = np.stack(kf_stack)
graph_stack = np.stack(graph_stack)
rand_stack = np.stack(rand_stack)


def compute_mse_and_se(estimates, truths):
    errors = np.linalg.norm(estimates - truths, axis=2)**2
    mse = np.mean(errors, axis=0)
    se = np.std(errors, axis=0) / np.sqrt(errors.shape[0])
    return mse, se


mse_obs, se_obs = compute_mse_and_se(obs_stack, true_stack)
mse_approx, se_approx = compute_mse_and_se(approx_stack, true_stack)
mse_kf, se_kf = compute_mse_and_se(kf_stack, true_stack)
mse_graph, se_graph = compute_mse_and_se(graph_stack, true_stack)
mse_rand, se_rand = compute_mse_and_se(rand_stack, true_stack)

plt.figure(figsize=(12, 8))
plt.plot(time, mse_obs, '--', label='Observations')
plt.fill_between(time, mse_obs - se_obs, mse_obs + se_obs, alpha=0.2)
plt.plot(time, mse_kf, ':', label='Kalman Filter')
plt.fill_between(time, mse_kf - se_kf, mse_kf + se_kf, alpha=0.2)
plt.plot(time, mse_rand, '-', label='Random Search')
plt.fill_between(time, mse_rand - se_rand, mse_rand + se_rand, alpha=0.2)
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
plt.savefig("MSE_over_time_non_linear.png", dpi=1200)
plt.savefig("MSE_over_time_non_linear.pdf", dpi=1200)
plt.show()
