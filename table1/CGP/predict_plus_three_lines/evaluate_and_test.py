import os
import re
import numpy as np
import gp  # Your custom GP module
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
g.n = 10
g.o = 5
g.a = 2
g.p = 0
g.lmb = 1000


def from_graphviz(g, dot_str):
    import re
    import numpy as np

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

    # Build input map preserving order of appearance
    input_map = {}
    for src, tgt in edges:
        input_map.setdefault(tgt, []).append(src)

    all_node_ids = set(node_labels.keys()) | {src for src, tgt in edges} | {tgt for src, tgt in edges}
    max_node_id = max(all_node_ids)
    total_rows = max(max_node_id + 1, g.i + g.n + g.o)

    gen = np.zeros((total_rows, 1 + g.a + g.p), dtype=int)
    node_id_to_row = {}

    # Inputs
    for nid, label in node_labels.items():
        if label.startswith("i") and label[1:].isdigit():
            idx = int(label[1:])
            node_id_to_row[nid] = idx

    # Functional nodes
    for nid in range(total_rows):
        label = node_labels.get(nid)
        if label in g.names:
            op_idx = g.names.index(label)
            gen[nid, 0] = op_idx
            node_id_to_row[nid] = nid

            # Use exact order from edges
            inputs = input_map.get(nid, [])
            for j in range(min(g.a, len(inputs))):
                gen[nid, 1 + j] = node_id_to_row.get(inputs[j], inputs[j])

        elif label is None:
            continue

        elif not label.startswith("i") and not label.startswith("o"):
            # Default unknowns to 'add' with dummy inputs
            gen[nid, 0] = g.names.index("add")
            gen[nid, 1:] = 0
            node_labels[nid] = "add"

    # Outputs
    output_base = g.i + g.n  # Output nodes start after input and function nodes
    for o in range(g.o):
        for nid, label in node_labels.items():
            if label == f"o{o}":
                srcs = input_map.get(nid, [])
                if srcs:
                    out_idx = output_base + o
                    gen[out_idx, 0] = 0  # identity passthrough (assumed safe default)
                    gen[out_idx, 1] = node_id_to_row.get(srcs[0], srcs[0])

    return gen







def normalize_dot(dot):
    return "\n".join(sorted(line.strip() for line in dot.strip().splitlines() if line.strip()))


# === FILE PARSING ===
pattern = re.compile(
    r"(digraph\s*\{.*?\})",
    re.DOTALL
)

log_dir = "./"
total_graphs = 0
digraphs = []

for root, _, files in os.walk(log_dir):
    for file in files:
        if file.endswith(".out"):
            path = os.path.join(root, file)
            print(f"\n📄 Reading: {path}")
            with open(path) as f:
                content = f.read()
                matches = pattern.findall(content)

                if not matches:
                    print(f"⚠️ No Graphviz match found in: {file}")
                    continue

                for dot in matches:
                    print(f"\n--- Evaluating graph from {file} ---")
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
                        digraphs.append((path, predict))
                        total_graphs += 1
                    except Exception as e:
                        print(f"❌ Error parsing/evaluating graph: {e}")

print(f"\n✅ Done. Total evaluated graphs: {total_graphs}")




#print("digraphs : ",digraphs)
#exit(1)
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
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y = z - (self.H @ self.x)
        self.S = (self.H @ (self.P @ self.H.T)) + self.R
        self.K = ((self.P @ self.H.T) @ np.linalg.inv(self.S))
        self.x = self.x + (self.K @ self.y)
        I = np.eye(self.F.shape[0])
        self.P = ((I - (self.K @ self.H)) @ self.P)
        return self.x


def execute(gen, x):
    return gp.execute(g, gen, x)

def distance_from_target_function(predict, traj):
    xx = np.array([0, 0], dtype=float)
    mse_true_trajectory = 0.0
    P = np.eye(dim)
    diff = []
    try:
        rn = gp.reachable_nodes(g, predict)
    except Exception:
        return float('inf')

    kf = KalmanFilter(F, B, H, Q, R, P, x=np.array([0, 0], dtype=float))

    for x, z in traj:
        try:
            xp, P, y, S,  K  = execute(predict, [xx, F, P, Q, z, R])
            if xp.shape != (dim,) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isnan(P)) or \
               np.any(np.isinf(xp)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')

            #y = z - xp
            #S = H @ (P @ H.T) + R
            #K = (P) @ np.linalg.inv(S)
            xx = xp + (K @ y)
            I = np.eye(dim)
            P = (I - (K @ H)) @ P

            x_true = kf.predict(u)
            kf.update(z)

            if x_true.shape != xp.shape:
                return float('inf')
            
            diff_current = x - xx
            if np.any(np.isinf(diff_current)) or np.any(np.isnan(diff_current)) or x.shape != xx.shape:
                return float('inf')
            diff.append(diff_current @ diff_current.T)

        except (ValueError, TypeError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
            return float('inf')

    loss = np.mean(diff)
    if loss < 0 :
        return float('inf')
    return loss if not math.isnan(loss) else float('inf')


def distance_from_kalman_filter(predict, traj):
    xx = np.array([0, 0], dtype=float)
    mse_true_trajectory = 0.0
    P = np.eye(dim)
    diff = []
    try:
        rn = gp.reachable_nodes(g, predict)
    except Exception:
        return float('inf')
    x_est = np.array([0.0, 0.0])

    kf = KalmanFilter(F, B, H, Q, R, P, x=np.array([0, 0], dtype=float))

    for x, z in traj:
        try:
            xp, P, y, S,  K, x_est  = execute(predict, [x_est, F, P, Q, z, R])
            if xp.shape != (dim,) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isnan(P)) or \
               np.any(np.isinf(xp)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')

            # xp = z
            # P = R
            # y =  F@xx - z
            # S = ((self.F @ self.P) @ self.F.T) + self.Q + R
            #
            #
            #y = z - xp
            #S = H @ (P @ H.T) + R

            # xp = z
            # P = R
            # y =  F@xx - z
            # S = ((self.F @ self.P) @ self.F.T) + self.Q + R
            #K = (P @ H.T) @ np.linalg.inv(S)
            #xx = xp + (K @ y)
            #I = np.eye(dim)
            P = (I - (K @ H)) @ P

            x_true = kf.predict(u)
            x_kalman = kf.update(z)

            if x_true.shape != xp.shape:
                return float('inf')
            print("x = ",x,"x_est = ",x_est)
            diff_current = x - x_est
            if np.any(np.isinf(diff_current)) or np.any(np.isnan(diff_current)):
                return float('inf')
            diff.append(diff_current @ diff_current.T)

        except (ValueError, TypeError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
            return float('inf')
    print("loss = ",loss)
    loss = np.mean(diff)
    return loss if not math.isnan(loss) else float('inf')



# Assuming all setup and imports from your script are done
def generate_trajectory(length=200, seed=None):
    rng = np.random.default_rng(seed)
    x = np.array([0, 0], dtype=float)
    traj = []
    for _ in range(length):
        x = F @ x + cQ @ rng.normal(0, 1, dim)
        z = H @ x + cR @ rng.normal(0, 1, dim)
        traj.append((x.copy(), z.copy()))
    return traj




g.nodes = (matmul, minus, add, transpose, inv)
g.names = ("matmul", "minus", "add", "transpose", "inv")
g.arity = (2, 2, 2, 1, 1)
g.args = (0, 0, 0, 0, 0)

g.i = 6
g.n = 10
g.o = 5
g.a = 2
g.p = 0
g.lmb = 1000




predict0 = gp.build(
        g,
        #  0     1    2    3   4    5    6        7        8           9        10       11     12    13   14        15   16   17   18   19  
        ["i0", "i1","i2","i3","i4","i5","matmul","matmul", "transpose","matmul", "add","minus","add","inv","matmul","o0","o1","o2","o3","o4"],#
        [
            (1, 6), # x = (F @ xx)
            (0, 6),
            (1, 7), # (F @ P)
            (2, 7),
            (1, 8), # F.T
            (7, 9), # ((F @ P) @ F.T)
            (8, 9),
            (9, 10), # P = ((F @ P) @ F.T) + Q 
            (3, 10),
            (4, 11), # y = z - self.x
            (6, 11),
            (10, 12), # S = R + P
            (5, 12),
            (12,13),  # K = (P) @ np.linalg.inv(S)         
            (10,14),
            (13,14),
            (6, 15),
            (10, 16),
            (11, 17),
            (12, 18),
            (14, 19),
        ],  # o1
        [])


N = 100
dim = 2
dt = 0.1
x = np.array([0, 0], dtype=float)
u = np.array([0, 0], dtype=float)
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
I = np.eye(dim)
B = np.eye(dim)

validation_trajectories = []
all_scores = []
for trial in range(50):
    traj = generate_trajectory(length=500, seed=12 + trial)
    validation_trajectories.append(traj)

print("here")


best_score = float('inf')
best_predict = None
seen_hashes = set()
i = 0
for filename, predict in digraphs:
    key = predict.tobytes()
    if key in seen_hashes:
        #print(f"Skipping already seen graph: {filename}")
        continue
    seen_hashes.add(key)    
    try:
        all_scores = []
        for trajectory in validation_trajectories:
            #score = distance_from_kalman_filter(predict,trajectory)
            score = distance_from_target_function(predict,trajectory)
            all_scores.append(score)
        score = np.mean(all_scores)
        #score = distance_from_kalman_filter(predict)
        sys.stdout.write(f"{filename} --> Score: {score:.6f}"+"\n")
        sys.stdout.flush()

        if score < best_score:
            best_score = score
            best_predict = predict

    except Exception as e:
        print(f"Evaluation failed for {filename}: {e}")

if best_predict is not None:
    sys.stdout.write(f"\n🏆 Best Graph Score: {best_score}"+"\n")
    sys.stdout.write(f"🏁 Best Graphviz:\n{gp.as_graphviz(g, best_predict)}"+"\n")
    all_scores =[]
    for trajectory in validation_trajectories:
        #score = distance_from_kalman_filter(predict,trajectory)
        score = distance_from_target_function(best_predict,trajectory)
        all_scores.append(score)
    score = np.mean(all_scores)

    #score = distance_from_kalman_filter(predict)
    #sys.stdout.write(f"Distance Score of Kalman Filter: {score:.6f}"+"\n")
    #sys.stdout.flush()


test_trajectories = []
mse_real_trajectories = []
mses_kalman_filter = []
for trial in range(50):
    traj = generate_trajectory(length=500, seed=32 + trial)
    test_trajectories.append(traj)




# Evaluate best_predict on test trajectories
mse_best_predict = []

for trajectory in test_trajectories:
    mse = distance_from_target_function(best_predict, trajectory)
    mse_best_predict.append(mse)

mse_best_predict = np.array(mse_best_predict)
mean_best = np.mean(mse_best_predict)
stderr_best = np.std(mse_best_predict, ddof=1) / np.sqrt(len(mse_best_predict))

sys.stdout.write("Evolved predictor performance\n")
sys.stdout.write(f"MSE on test set (best_predict)        : {mean_best:.6f} ± {stderr_best:.6f}\n")
sys.stdout.flush()

# Now evaluate Kalman filter on the same test set
mse_real_trajectories = []
mses_kalman_filter = []

for trajectory in test_trajectories:
    mse_real = distance_from_target_function(kf, trajectory)
    mse_kalman = distance_from_kalman_filter(kf, trajectory)
    mse_real_trajectories.append(mse_real)
    mses_kalman_filter.append(mse_kalman)

mse_real_trajectories = np.array(mse_real_trajectories)
mses_kalman_filter = np.array(mses_kalman_filter)

mean_real = np.mean(mse_real_trajectories)
stderr_real = np.std(mse_real_trajectories, ddof=1) / np.sqrt(len(mse_real_trajectories))

mean_kalman = np.mean(mses_kalman_filter)
stderr_kalman = np.std(mses_kalman_filter, ddof=1) / np.sqrt(len(mses_kalman_filter))

sys.stdout.write("\nKalman filter performance\n")
sys.stdout.write(f"MSE real trajectories on test set     : {mean_real:.6f} ± {stderr_real:.6f}\n")
sys.stdout.write(f"MSE Kalman filter on test set         : {mean_kalman:.6f} ± {stderr_kalman:.6f}\n")
sys.stdout.flush()
