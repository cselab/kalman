import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18
})

# === Graph structure (unchanged) ===
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

best_graph = '''
digraph {
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

def from_graphviz(g, dot_str):
    import re
    edges = []
    node_labels = {}
    for line in dot_str.splitlines():
        line = line.strip()
        if not line or line.startswith("digraph") or line in {"{", "}"}:
            continue
        label_match = re.match(r'^(\d+)\s+\[label\s*=\s*"?([^"]+?)"?\]', line)
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

best_graph = from_graphviz(g, best_graph)

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

# === Simulate ===
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
nprng = np.random.default_rng(seed=12345)
x = np.array([0, 0], dtype=float)
P = np.eye(dim)

true_states = []
observations = []
approx_states = []
kf_states = []

kf = KalmanFilter(F, B, H, Q, R, np.eye(dim), x.copy())
x_approx = np.array([0, 0], dtype=float)

true_states.append(x.copy())
observations.append(x.copy())
approx_states.append(x.copy())
kf_states.append(x.copy())
for i in range(10):
    x = F @ x + cQ @ abs(nprng.normal(0, 1, dim))
    z = H @ x + cR @ abs(nprng.normal(0, 1, dim))
    _, P, _, _, _, x_approx = our_approximation(x_approx.copy(), F, P, Q, z, R)
    kf.predict()
    kf_state = kf.update(z)

    true_states.append(x.copy())
    observations.append(z.copy())
    approx_states.append(x_approx.copy())
    kf_states.append(kf_state.copy())

true_states = np.array(true_states)
observations = np.array(observations)
approx_states = np.array(approx_states)
kf_states = np.array(kf_states)

# === Plot: Trajectory ===
plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0], true_states[:, 1], '*-', label='True Trajectory', linewidth=1.5, markersize=10)
plt.plot(observations[:, 0], observations[:, 1], '*', alpha=0.4, label='Observations', markersize=10)
plt.plot(approx_states[:, 0], approx_states[:, 1], '*--', label='Our Approximation', markersize=10)
plt.plot(kf_states[:, 0], kf_states[:, 1], '*:', label='Kalman Filter', markersize=10)



plt.legend()
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_comparison.png", dpi=1200)
plt.savefig("trajectory_comparison.pdf", dpi=1200)

plt.show()

# === Plot: Position Over Time ===
time = np.arange(len(true_states))
plt.figure(figsize=(12, 4))
plt.plot(time, true_states[:, 0], '*-', label="True Position")
plt.plot(time, observations[:, 0], '*', alpha=0.5, label="Observed Position")
plt.plot(time, approx_states[:, 0], '*--', label="Approximation")
plt.plot(time, kf_states[:, 0], '*:', label="Kalman Filter")
plt.title("Position Over Time")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("position_over_time.png", dpi=1200)
plt.show()

# === Plot: Velocity Over Time ===
plt.figure(figsize=(12, 4))
plt.plot(time, true_states[:, 1], '*-', label="True Velocity")
plt.plot(time, observations[:, 1], '*', alpha=0.5, label="Observed Velocity")
plt.plot(time, approx_states[:, 1], '*--', label="Approximation")
plt.plot(time, kf_states[:, 1], '*:', label="Kalman Filter")
plt.title("Velocity Over Time")
plt.xlabel("Time Step")
plt.ylabel("Velocity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("velocity_over_time.png", dpi=1200)
plt.show()
