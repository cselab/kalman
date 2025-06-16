import os
import re
import numpy as np
import gp  # Assumes your custom GP module is available
import sys
import math

# === GP CONFIGURATION ===
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

kalman_filter = gp.build(
        g,
        ["i0", "i1", "i2", "i3", "i4", "i5", "matmul", "matmul", "transpose", "matmul",
         "add", "minus", "add", "inv", "matmul", "matmul", "add", "matmul", "minus", "o0",
         "o1", "o2", "o3", "o4", "o5"],
        [(1, 6), (0, 6), (1, 7), (2, 7), (1, 8), (7, 9), (8, 9), (9, 10), (3, 10),
         (4, 11), (6, 11), (10, 12), (5, 12), (12, 13), (10, 14), (13, 14), (14, 15),
         (11, 15), (6, 16), (15, 16), (14, 17), (10, 17), (10, 18), (17, 18), (6, 19),
         (18, 20), (11, 21), (12, 22), (14, 23), (16, 24)],
        [])

def from_graphviz(g, dot_str):
    import re
    import numpy as np

    edges = []
    node_labels = {}

    # Parse node and edge lines
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

    # Build input map
    input_map = {}
    for src, tgt in edges:
        input_map.setdefault(tgt, []).append(src)

    max_node_id = max(node_labels.keys())
    total_rows = max(max_node_id + 1, g.i + g.n + g.o)

    # Allocate genotype matrix (long enough to preserve node IDs)
    gen = np.zeros((total_rows, 1 + g.a + g.p), dtype=int)
    node_id_to_row = {}

    # Map inputs
    for nid, label in node_labels.items():
        if label.startswith("i") and not label.startswith("inv"):
            idx = int(label[1:])
            node_id_to_row[nid] = idx

    # Functional nodes
    for nid in range(total_rows):
        label = node_labels.get(nid)
        if label in g.names:
            op_idx = g.names.index(label)
            gen[nid, 0] = op_idx
            node_id_to_row[nid] = nid

            # Assign inputs (resolve their mapped rows)
            inputs = input_map.get(nid, [])
            for j in range(min(g.a, len(inputs))):
                gen[nid, 1 + j] = node_id_to_row.get(inputs[j], inputs[j])
        elif label is None:
            continue
        elif not label.startswith("i") and not label.startswith("o")  and not label.startswith("inv"):
            # fill dummy "add" node
            gen[nid, 0] = g.names.index("add")
            gen[nid, 1:] = 0
            node_labels[nid] = "add"

    # Outputs
    for o in range(g.o):
        for nid, label in node_labels.items():
            if label == f"o{o}":
                srcs = input_map.get(nid, [])
                if srcs:
                    gen[nid, 0] = 0  # dummy op for output
                    gen[nid, 1] = node_id_to_row.get(srcs[0], srcs[0])

    return gen




def normalize_dot(dot):
    return "\n".join(sorted(line.strip() for line in dot.strip().splitlines() if line.strip()))

def get_F_Q(effective_dt):
    F = np.array([[1, effective_dt], [0, 1]], dtype=float)
    G = np.array([[0.5 * effective_dt ** 2], [effective_dt]])
    Q = G @ G.T
    return F, Q

def generate_trajectory(length=500, seed=42):
    dim = 2
    dt = 1.0
    cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
    cR = np.eye(dim)
    H = np.eye(dim)
    R = cR @ cR.T
    B = np.eye(dim)
    x = np.array([0, 0], dtype=float)
    rng = np.random.default_rng(seed)
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
    return traj

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
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x

def execute(gen, x):
    return gp.execute(g, gen, x)

def distance_from_target_function(predict, traj, alpha=1.0):
    dim = 2
    x_est = np.zeros(dim)
    P = np.eye(dim)
    squared_errors, pred_errors = [], []

    try:
        gp.reachable_nodes(g, predict)
    except:
        print(0)
        return float('inf')

    for x_true, z, F_dyn, Q_dyn in traj:
        try:
            xp, P, y, S, K, x_est = execute(predict, [x_est, F_dyn, P, Q_dyn, z, R])
            if xp.shape != (dim,) or np.any(~np.isfinite(xp)):
                print(1)
                return float('inf')
            pred_errors.append(np.dot(x_true - xp, x_true - xp))
            squared_errors.append(np.dot(x_true - x_est, x_true - x_est))
        except Exception as e:
            print(2,)
            print(e)
            return float('inf')

    return alpha * np.mean(squared_errors) + (1 - alpha) * np.mean(pred_errors)

def distance_from_kalman_filter(predict, traj):
    dim = 2
    x_est = np.zeros(dim)
    P = np.eye(dim)
    diff = []

    try:
        gp.reachable_nodes(g, predict)
    except:
        return float('inf')

    for x_true, z, F_dyn, Q_dyn in traj:
        try:
            xp, P, y, S, K, x_est = execute(predict, [x_est, F_dyn, P, Q_dyn, z, R])
            kf = KalmanFilter(F_dyn, np.eye(dim), np.eye(dim), Q_dyn, R, P.copy(), x_est.copy())
            x_kalman = kf.predict()
            kf.update(z)
            err = xp - x_kalman
            if np.any(~np.isfinite(err)):
                print(10)
                return float('inf')
            diff.append(np.dot(err, err))
        except:
            print(20)
            return float('inf')

    return np.mean(diff) if diff else float('inf')

# === EVALUATION ===
pattern = re.compile(r"Generation\s+(\d+)\s+-\s+Island\s+(\d+).*?BEST Graph\s+:\s+(digraph \{.*?\})", re.DOTALL)
pattern = re.compile(r"Generation\s+(\d+)\s+-\s+Island\s+(\d+).*?BEST Graph\s+:\s+(digraph \{.*?\})", re.DOTALL)
log_dir = "./"
total_graphs = 0
digraphs = []
for root, _, files in os.walk(log_dir):
    for file in files:
        if file.endswith(".out"):
            path = os.path.join(root, file)
            print(f"\n📄 Reading: {path}")
            with open(path) as f:
                matches = pattern.findall(f.read())

                for gen, island, dot in matches:
                    print(f"\n--- Evaluating gen{gen}_island{island} ---")
                    try:
                        predict = from_graphviz(g, dot)

                        regenerated = gp.as_graphviz(g, predict)

                        if normalize_dot(dot) != normalize_dot(regenerated):
                            print("❌ MISMATCH between input DOT and regenerated Graphviz!")
                            print("\n--- Original DOT ---\n", dot)
                            print("\n--- Regenerated DOT ---\n", regenerated)
                            raise SystemExit("❌ Stopped due to mismatch.")
                        else:
                            print("✅ DOT matches regenerated Graphviz.")
                        digraphs.append((path,predict))
                        total_graphs += 1
                    except Exception as e:
                        print(f"❌ Error parsing/evaluating graph: {e}")

print(f"\n✅ Done. Total evaluated graphs: {total_graphs}")
total_graphs = 0


# === CONTINUATION OF EVALUATION ===
print(f"\n✅ Done. Total evaluated graphs: {total_graphs}")


# === Trajectory Generation with Small Delays ===
dim = 2
dt = 1.0
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
cR = np.eye(dim)
H = np.eye(dim)
R = cR @ cR.T
B = np.eye(dim)
x = np.array([0, 0], dtype=float)
traj = []
true_states = [x.copy()]
nprng = np.random.default_rng(seed=12345)

validation_trajectories = [generate_trajectory(seed=12 + i) for i in range(50)]
test_trajectories = [generate_trajectory(seed=32 + i) for i in range(50)]

best_score = float('inf')
best_predict = None
seen_hashes = set()


# Kalman filter score

val_scores = [distance_from_target_function(kalman_filter, traj) for traj in validation_trajectories]
score = np.mean(val_scores)
print("kalman filter validation score :",score)

val_scores = [distance_from_target_function(kalman_filter, traj) for traj in test_trajectories]
score = np.mean(val_scores)
print("kalman filter test score :",score)


for filename, predict in digraphs:
    key = predict.tobytes()
    if key in seen_hashes:
        continue
    seen_hashes.add(key)
    try:
        val_scores = [distance_from_target_function(predict, traj) for traj in validation_trajectories]
        score = np.mean(val_scores)
        print(f"{filename} --> Score: {score:.6f}")
        if score < best_score:
            best_score = score
            best_predict = predict
    except Exception as e:
        print(f"Evaluation failed for {filename}: {e}")

if best_predict is not None:
    print(f"\n🏆 Best Graph Score: {best_score}")
    print(f"🏁 Best Graphviz:\n{gp.as_graphviz(g, best_predict)}")

    traj_mses = [distance_from_target_function(best_predict, traj) for traj in test_trajectories]

    print(f"MSE real trajectories on test set: {np.mean(traj_mses):.6f}")
