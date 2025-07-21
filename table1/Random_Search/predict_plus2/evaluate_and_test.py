import os
import re
import numpy as np
import gp
import sys
import math

# === GP CONFIGURATION ===
class g:
    pass

def minus(inp, args): return inp[0] - inp[1]
def matmul(inp, args): return inp[0] @ inp[1]
def add(inp, args): return inp[0] + inp[1]
def transpose(inp, args): return inp[0].T

g.nodes = (matmul, minus, add, transpose)
g.names = ("matmul", "minus", "add", "transpose")
g.arity = (2, 2, 2, 1)
g.args = (0, 0, 0, 0)
g.i = 6     # [x_est, F, P, Q, z, R]
g.n = 7
g.o = 4     # [xp, P, y, S]
g.a = 2
g.p = 0
g.lmb = 1000

alpha = 1.0  # only post-update MSE is used

def from_graphviz(g, dot_str):
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

    total_rows = g.i + g.n + g.o
    gen = np.zeros((total_rows, 1 + g.a + g.p), dtype=int)
    node_to_row = {}

    for nid, label in node_labels.items():
        if label.startswith("i") and label[1:].isdigit():
            row = int(label[1:])
            node_to_row[nid] = row

    current_row = g.i

    for nid in sorted(node_labels):
        label = node_labels[nid]
        if label in g.names:
            if current_row >= g.i + g.n:
                continue
            op_idx = g.names.index(label)
            gen[current_row, 0] = op_idx
            inputs = input_map.get(nid, [])
            for j in range(min(len(inputs), g.a)):
                src_nid = inputs[j]
                if src_nid not in node_to_row:
                    raise ValueError(f"Missing source node mapping for {src_nid}")
                gen[current_row, 1 + j] = node_to_row[src_nid]
            node_to_row[nid] = current_row
            current_row += 1

    output_base = g.i + g.n
    for nid, label in node_labels.items():
        if label.startswith("o") and label[1:].isdigit():
            oidx = int(label[1:])
            if oidx >= g.o:
                continue
            srcs = input_map.get(nid, [])
            if not srcs:
                continue
            src = srcs[0]
            if src not in node_to_row:
                raise ValueError(f"Missing source for output node {nid}")
            gen[output_base + oidx, 0] = 0
            gen[output_base + oidx, 1] = node_to_row[src]

    return gen

def graphs_structurally_equivalent(dot1, dot2):
    def extract_structure(dot):
        node_types = {}
        edges = set()
        for line in dot.splitlines():
            line = line.strip()
            if not line or line.startswith("digraph") or line in {"{", "}"}:
                continue
            label_match = re.match(r'^(\d+)\s+\[label\s*=\s*"?([^"]+?)"?\]', line)
            edge_match = re.match(r'^(\d+)\s*->\s*(\d+)', line)
            if label_match:
                nid, label = label_match.groups()
                node_types[nid] = label
            elif edge_match:
                src, tgt = edge_match.groups()
                edges.add((node_types.get(src, src), node_types.get(tgt, tgt)))
        return frozenset(node_types.values()), frozenset(edges)

    return extract_structure(dot1) == extract_structure(dot2)

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x = x.copy()
    def predict(self, u=np.zeros((1,1))):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    def update(self, z):
        self.y = z - self.x
        self.S = self.P + self.R
        self.K = self.P @ np.linalg.inv(self.S)
        self.x = self.x + self.K @ self.y
        self.P = self.P - self.K @ self.P
        return self.x

def execute(gen, x):
    return gp.execute(g, gen, x)

def distance_from_target_function(predict, traj):
    x_est = np.array([0.0, 0.0])
    P = np.eye(dim)
    squared_errors = []

    try:
        _ = gp.reachable_nodes(g, predict)
    except Exception:
        return float('inf')

    for x_true, z in traj:
        try:
            result = execute(predict, [x_est, F, P, Q, z, R])
            if not isinstance(result, (list, tuple)) or len(result) != 4:
                return float('inf')
            xp, P_pred, y, S = result

            if xp.shape != (dim,) or P_pred.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isinf(xp)) or \
               np.any(np.isnan(P_pred)) or np.any(np.isinf(P_pred)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P_pred) > 1e6:
                return float('inf')

            K = (P_pred @ H.T) @ np.linalg.inv(S)
            x_est = xp + (K @ y)
            P = (np.eye(dim) - (K @ H)) @ P_pred

            error = x_true - x_est
            if error.shape != (dim,) or np.any(np.isnan(error)) or np.any(np.isinf(error)):
                return float('inf')

            squared_errors.append(np.dot(error, error))

        except Exception:
            return float('inf')

    if not squared_errors:
        return float('inf')

    return np.mean(squared_errors)

def kalman_baseline(traj):
    kf = KalmanFilter(F, B, H, Q, R, np.eye(2), x=np.zeros(2))
    errors = []
    for x, z in traj:
        kf.predict(u)
        x_est = kf.update(z)
        errors.append((x - x_est) @ (x - x_est))
    return np.mean(errors)

dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
R = np.eye(dim)
H = np.eye(dim)
B = np.eye(dim)
u = np.zeros(dim)

def generate_trajectory(length=500, seed=None):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    traj = []
    for _ in range(length):
        x = F @ x + cQ @ rng.normal(0, 1, dim)
        z = H @ x + R @ rng.normal(0, 1, dim)
        traj.append((x.copy(), z.copy()))
    return traj

pattern = re.compile(r"\[\d+\]\s+New best score:\s+[\d.eE+-]+\s+Graph:\s+(digraph\s*\{.*?\})", re.DOTALL)
log_dir = "./"
digraphs = []

for root, _, files in os.walk(log_dir):
    for file in files:
        if file.endswith(".out"):
            path = os.path.join(root, file)
            with open(path) as f:
                content = f.read()
                matches = pattern.findall(content)
                for dot in matches:
                    print(f"\n📄 Checking {file}")
                    try:
                        predict = from_graphviz(g, dot)
                        regenerated = gp.as_graphviz(g, predict)
                        if not graphs_structurally_equivalent(dot, regenerated):
                            print("❌ Structure mismatch detected!")
                            print("\n🔍 Original DOT:\n", dot)
                            print("\n🔁 Regenerated DOT:\n", regenerated)
                            print("⚠️ Skipping this graph and continuing...\n")
                            continue
                        else:
                            print("✅ DOT structure is equivalent.")
                        digraphs.append((path, predict))
                    except Exception as e:
                        print("❌ Parse error:", e)

print(f"\n✅ Total loaded graphs: {len(digraphs)}")

validation_trajectories = [generate_trajectory(seed=12+i) for i in range(50)]
best_score = float('inf')
best_predict = None
seen_hashes = set()

for filename, predict in digraphs:
    key = predict.tobytes()
    if key in seen_hashes:
        continue
    seen_hashes.add(key)
    scores = [distance_from_target_function(predict, traj) for traj in validation_trajectories]
    score = np.mean(scores)
    print(f"{filename} --> Score: {score:.6f}")
    if score < best_score:
        best_score = score
        best_predict = predict

if best_predict is not None:
    print(f"\n🏆 Best Score: {best_score:.6f}")
    print(f"\n📈 Best Graphviz:\n{gp.as_graphviz(g, best_predict)}")

    test_trajectories = [generate_trajectory(seed=32+i) for i in range(50)]
    pred_scores = [distance_from_target_function(best_predict, traj) for traj in test_trajectories]
    kalman_scores = [kalman_baseline(traj) for traj in test_trajectories]

    mean_pred = np.mean(pred_scores)
    stderr_pred = np.std(pred_scores, ddof=1) / np.sqrt(len(pred_scores))
    mean_kf = np.mean(kalman_scores)
    stderr_kf = np.std(kalman_scores, ddof=1) / np.sqrt(len(kalman_scores))

    print("\n📊 Final Test Results")
    print(f"Evolved Predictor MSE     : {mean_pred:.6f} ± {stderr_pred:.6f}")
    print(f"Kalman Filter MSE         : {mean_kf:.6f} ± {stderr_kf:.6f}")
