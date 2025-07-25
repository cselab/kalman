import os
import re
import math
import numpy as np
import gp

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
g.i = 5     # [xx, F, P, Q, z]
g.n = 7
g.o = 3     # [xp, P, y]
g.a = 2
g.p = 0
g.lmb = 1000

def from_graphviz(g, dot_str):
    edges = []
    node_labels = {}

    for line in dot_str.splitlines():
        line = line.strip()
        if not line or line.startswith("digraph") or line in {"{", "}"}:
            continue

        label_match = re.match(r'^(\d+)\s+\[label\s*=\s*"?([^"\]]+?)"?\]', line)
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
            idx = int(label[1:])
            node_to_row[nid] = idx

    fn_row = g.i
    for nid, label in sorted(node_labels.items()):
        if label in g.names:
            if fn_row >= g.i + g.n:
                break
            op_idx = g.names.index(label)
            gen[fn_row, 0] = op_idx
            inputs = input_map.get(nid, [])
            for j in range(min(len(inputs), g.a)):
                gen[fn_row, 1 + j] = node_to_row.get(inputs[j], inputs[j])
            node_to_row[nid] = fn_row
            fn_row += 1

    output_base = g.i + g.n
    for nid, label in node_labels.items():
        if label.startswith("o") and label[1:].isdigit():
            oidx = int(label[1:])
            if oidx >= g.o:
                continue
            inputs = input_map.get(nid, [])
            if not inputs:
                continue
            src = inputs[0]
            mapped = node_to_row.get(src, src)
            gen[output_base + oidx, 0] = 0
            gen[output_base + oidx, 1] = mapped

    return gen

def graphs_structurally_equivalent(dot1, dot2):
    def extract_structure(dot):
        node_types = {}
        edges = set()

        for line in dot.splitlines():
            line = line.strip()
            if not line or line.startswith("digraph") or line in {"{", "}"}:
                continue
            label_match = re.match(r'^(\d+)\s+\[label\s*=\s*"?([^"\]]+?)"?\]', line)
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
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.P
        return self.x

def execute(gen, x):
    return gp.execute(g, gen, x)

def distance_from_target_function(predict, traj):
    xx = np.array([0, 0], dtype=float)
    P = np.eye(dim)
    diff = []

    try:
        gp.reachable_nodes(g, predict)
    except Exception:
        return float('inf')

    kf = KalmanFilter(F, B, H, Q, R, P, x=np.array([0, 0], dtype=float))

    for x, z in traj:
        try:
            xp, P_pred, y = execute(predict, [xx, F, P, Q, z])
            if xp.shape != (dim,) or P_pred.shape != (dim, dim):
                return float('inf')
            if not np.all(np.isfinite(xp)) or not np.all(np.isfinite(P_pred)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P_pred) > 1e6:
                return float('inf')

            S = H @ (P_pred @ H.T) + R
            K = (P_pred @ H.T) @ np.linalg.inv(S)
            xx = xp + (K @ y)
            P = (np.eye(dim) - (K @ H)) @ P_pred

            x_true = kf.predict(u)
            kf.update(z)

            if np.linalg.norm(xx) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')
            if x_true.shape != xp.shape:
                return float('inf')

            diff_current = x - xx
            if not np.all(np.isfinite(diff_current)):
                return float('inf')

            diff.append(diff_current @ diff_current.T)

        except (ValueError, TypeError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
            return float('inf')

    loss = np.mean(diff)
    return loss if not math.isnan(loss) else float('inf')

def kalman_baseline(traj):
    kf = KalmanFilter(F, B, H, Q, R, np.eye(dim), x=np.zeros(dim))
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

pattern = re.compile(
    r"Island\s+[0-4]\s+BEST\s+Graph\s+:\s*(digraph\s*\{.*?\})",
    re.DOTALL
)

log_dir = "./"
digraphs = []

for root, _, files in os.walk(log_dir):
    for file in files:
        if file.endswith(".out"):
            path = os.path.join(root, file)
            with open(path) as f:
                content = f.read()
                matches = pattern.findall(content)
                for idx, dot in enumerate(matches):
                    print(f"\n📄 Checking {file} (graph {idx+1}/{len(matches)})")
                    try:
                        predict = from_graphviz(g, dot)
                        regenerated = gp.as_graphviz(g, predict)
                        if not graphs_structurally_equivalent(dot, regenerated):
                            print("❌ Structure mismatch detected!")
                            print("📥 Original DOT:\n", dot.strip())
                            print("📤 Regenerated DOT:\n", regenerated.strip())
                            print("⚠️ Skipping this graph and continuing...\n")
                            continue
                        print("✅ DOT structure is accepted.")
                        digraphs.append((path, predict))
                    except Exception as e:
                        print("❌ Parse error:", e)

print(f"\n✅ Total evaluated graphs: {len(digraphs)}")

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
