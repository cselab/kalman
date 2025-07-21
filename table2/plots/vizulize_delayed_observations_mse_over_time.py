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

def minus(inp, args): return inp[0] - inp[1]
def matmul(inp, args): return inp[0] @ inp[1]
def add(inp, args): return inp[0] + inp[1]
def transpose(inp, args): return inp[0].T
def inv(inp, args): return np.linalg.inv(inp[0])

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

best_graph_dot = """digraph {
  0 [label = i0]
  1 [label = i1]
  2 [label = i2]
  3 [label = i3]
  4 [label = i4]
  5 [label = i5]
  6 [label = "matmul"]
  1 -> 6 [color = red]
  4 -> 6 [color = blue]
  7 [label = "add"]
  2 -> 7 [color = red]
  5 -> 7 [color = blue]
  8 [label = "inv"]
  7 -> 8
  9 [label = "matmul"]
  5 -> 9 [color = red]
  8 -> 9 [color = blue]
  10 [label = "minus"]
  6 -> 10 [color = red]
  0 -> 10 [color = blue]
  11 [label = "transpose"]
  8 -> 11
  13 [label = "minus"]
  9 -> 13 [color = red]
  9 -> 13 [color = blue]
  14 [label = "add"]
  9 -> 14 [color = red]
  1 -> 14 [color = blue]
  16 [label = "matmul"]
  10 -> 16 [color = red]
  11 -> 16 [color = blue]
  18 [label = "add"]
  16 -> 18 [color = red]
  0 -> 18 [color = blue]
  22 [label = "matmul"]
  14 -> 22 [color = red]
  9 -> 22 [color = blue]
  24 [label = "minus"]
  11 -> 24 [color = red]
  13 -> 24 [color = blue]
  25 [label = o0]
  18 -> 25
  26 [label = o1]
  22 -> 26
  27 [label = o2]
  24 -> 27
  28 [label = o3]
  6 -> 28
  29 [label = o4]
  8 -> 29
  30 [label = o5]
  18 -> 30
}"""

random_search = """digraph {
  0 [label = i0]
  1 [label = i1]
  2 [label = i2]
  3 [label = i3]
  4 [label = i4]
  5 [label = i5]
  6 [label = "add"]
  3 -> 6 [color = red]
  2 -> 6 [color = blue]
  8 [label = "matmul"]
  4 -> 8 [color = red]
  6 -> 8 [color = blue]
  15 [label = "add"]
  6 -> 15 [color = red]
  1 -> 15 [color = blue]
  17 [label = "inv"]
  2 -> 17
  18 [label = "add"]
  1 -> 18 [color = red]
  5 -> 18 [color = blue]
  20 [label = "matmul"]
  8 -> 20 [color = red]
  17 -> 20 [color = blue]
  24 [label = "transpose"]
  8 -> 24
  25 [label = o0]
  0 -> 25
  26 [label = o1]
  15 -> 26
  27 [label = o2]
  0 -> 27
  28 [label = o3]
  18 -> 28
  29 [label = o4]
  24 -> 29
  30 [label = o5]
  20 -> 30
}"""



def from_graphviz(g, dot_str):
    import re
    edges = []
    node_labels = {}
    for line in dot_str.splitlines():
        line = line.strip()
        if not line or line.startswith("digraph") or line in {"{", "}"}:
            continue
        label_match = re.match(r'^(\d+)\s+\[label\s*=\s*\"?([^\"]+?)\"?\]', line)
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
        elif not label.startswith("i") and not label.startswith("o") and not label.startswith("inv"):
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

def kalman_filter(x, F, P, Q, z, R):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    y = z - x_pred
    S = P_pred + R
    K = P_pred @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(K)) - K) @ P_pred
    return x_upd, P_upd
#function_
def approximate(x, F, P, Q, z, R):
    a = F @ x
    b = F @ np.log(np.maximum(a * 0.03, 1e-8))
    c = F @ np.tanh(b * 0.2)
    xp = a + c
    P = F @ P @ F.T + 0.7 * Q
    y = z - xp
    S = P + 0.7 * R
    inv_S = np.linalg.inv(S)
    K = P @ inv_S + 0.15 * inv_S
    x = xp + K @ y
    x += 0.6 * F @ np.tanh(F @ x * 0.08)
    P = (np.eye(F.shape[0]) - K) @ P
    return x, P

def graph_approximate(x, F, P, Q, z, R):
    A = F @ R
    B = P + Q
    K = np.linalg.inv(B)
    S = Q @ K
    y = A - x
    T = K.T
    P = S - S
    F_ = S + F
    xp = y @ T + x
    S = F_ @ S
    K = T - P
    return xp, P, y, S, K, x

dim = 2
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)

def get_F_Q(effective_dt):
    F = np.array([[1, effective_dt], [0, 1]], dtype=float)
    G = np.array([[0.5 * effective_dt ** 2], [effective_dt]])
    Q = G @ G.T
    return F, Q

def generate_trajectory(length=501, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    traj = []
    true_states = [x.copy()]
    for t in range(1, length):
        delay = rng.uniform(0.01, 0.3)
        effective_dt = 1.0 + delay
        F_dyn, Q_dyn = get_F_Q(effective_dt)
        x = F_dyn @ x + cQ @ rng.normal(0, 1, 2)
        true_states.append(x.copy())
        ε_idx = t - delay
        t0 = int(np.floor(ε_idx))
        t1 = min(t0 + 1, len(true_states) - 1)
        α = ε_idx - t0
        x_interp = (1 - α) * true_states[t0] + α * true_states[t1]
        z = H @ x_interp + cR @ rng.normal(0, 1, 2)
        traj.append((x.copy(), z.copy(), F_dyn, Q_dyn))
    return traj, np.array(true_states)




best_graph = from_graphviz(g, best_graph_dot)
random_graph = from_graphviz(g, random_search)

n_runs = 1000
T = 500
time = np.arange(T)

all_mse_cgp = np.zeros((n_runs, T))
all_mse_kf = np.zeros((n_runs, T))
all_mse_approx = np.zeros((n_runs, T))
all_mse_obs = np.zeros((n_runs, T))
all_mse_random = np.zeros((n_runs, T))

for seed in range(n_runs):
    traj, true_states = generate_trajectory(seed=seed)
    x_est_cgp = x_est_kf = x_est_approx = x_est_random = np.zeros(2)
    P_cgp = P_kf = P_approx = P_random = np.eye(2)

    for t, (x_true, z, F, Q) in enumerate(traj):
        try:
            _, P_cgp, _, _, _, x_est_cgp = wavegp.execute(g, best_graph, [x_est_cgp.copy(), F.copy(), P_cgp.copy(), Q.copy(), z.copy(), R.copy()])
            all_mse_cgp[seed, t] = np.linalg.norm(x_est_cgp - x_true) ** 2
        except:
            all_mse_cgp[seed, t] = np.nan

        try:
            x_est_kf, P_kf = kalman_filter(x_est_kf, F, P_kf, Q, z, R)
            all_mse_kf[seed, t] = np.linalg.norm(x_est_kf - x_true) ** 2
        except:
            all_mse_kf[seed, t] = np.nan

        try:
            x_est_approx, P_approx = approximate(x_est_approx, F, P_approx, Q, z, R)
            all_mse_approx[seed, t] = np.linalg.norm(x_est_approx - x_true) ** 2
        except:
            all_mse_approx[seed, t] = np.nan

        try:
            _, P_random, _, _, _, x_est_random = wavegp.execute(g, random_graph, [x_est_random.copy(), F.copy(), P_random.copy(), Q.copy(), z.copy(), R.copy()])
            all_mse_random[seed, t] = np.linalg.norm(x_est_random - x_true) ** 2
        except:
            all_mse_random[seed, t] = np.nan

        all_mse_obs[seed, t] = np.linalg.norm(z - x_true) ** 2

mean_mse_graph = np.nanmean(all_mse_cgp, axis=0)
mean_mse_kf = np.nanmean(all_mse_kf, axis=0)
mean_mse_approx = np.nanmean(all_mse_approx, axis=0)
mean_mse_obs = np.nanmean(all_mse_obs, axis=0)
mean_mse_random = np.nanmean(all_mse_random, axis=0)
se_graph = np.nanstd(all_mse_cgp, axis=0) / np.sqrt(n_runs)
se_kf = np.nanstd(all_mse_kf, axis=0) / np.sqrt(n_runs)
se_approx = np.nanstd(all_mse_approx, axis=0) / np.sqrt(n_runs)
se_obs = np.nanstd(all_mse_obs, axis=0) / np.sqrt(n_runs)
se_random = np.nanstd(all_mse_random, axis=0) / np.sqrt(n_runs)

# Plot including Random CGP
plt.figure(figsize=(12, 8))
plt.plot(time, mean_mse_obs, '--', label='Observations')
plt.fill_between(time, mean_mse_obs - se_obs, mean_mse_obs + se_obs, alpha=0.2)
plt.plot(time, mean_mse_kf, ':', label='Kalman Filter')
plt.fill_between(time, mean_mse_kf - se_kf, mean_mse_kf + se_kf, alpha=0.2)
plt.plot(time, mean_mse_random, '--', label='Random Search')
plt.fill_between(time, mean_mse_random - se_random, mean_mse_random + se_random, alpha=0.2)
plt.plot(time, mean_mse_approx, '-.', label='Funsearch')
plt.fill_between(time, mean_mse_approx - se_approx, mean_mse_approx + se_approx, alpha=0.2)
plt.plot(time, mean_mse_graph, '-', label='CGP')
plt.fill_between(time, mean_mse_graph - se_graph, mean_mse_graph + se_graph, alpha=0.2)
plt.xlabel("Time Step")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("MSE_over_time_with_random_delay.png", dpi=1200)
plt.savefig("MSE_over_time_with_random_delay.pdf", dpi=1200)
plt.show()
