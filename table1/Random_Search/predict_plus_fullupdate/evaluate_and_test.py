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
g.n = 17
g.o = 6
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
            gen[nid, 0] = g.names.index("add")
            gen[nid, 1:] = 0
            node_labels[nid] = "add"

    # Outputs
    for o in range(g.o):
        for nid, label in node_labels.items():
            if label == f"o{o}":
                srcs = input_map.get(nid, [])
                if srcs:
                    gen[nid, 0] = 0
                    gen[nid, 1] = node_id_to_row.get(srcs[0], srcs[0])

    return gen







def normalize_dot(dot):
    return "\n".join(sorted(line.strip() for line in dot.strip().splitlines() if line.strip()))


# === FILE PARSING ===
pattern = re.compile(
    r"\[\d+\]\s+New best score:\s+[\d.eE+-]+\s+Graph:\s+(digraph\s*\{.*?\})",
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

def distance_from_target_function(predict, traj, alpha=1.0):
    """
    Computes a combined loss:
    loss = alpha * MSE(post-update) + (1 - alpha) * MSE(pre-update)

    Args:
        predict: the evolved function
        alpha (float): weight for post-update loss

    Returns:
        float: combined loss or inf on failure
    """
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

    x_est = np.array([0.0, 0.0])
    P = np.eye(dim)
    squared_errors = []
    pred_errors = []

    try:
        _ = gp.reachable_nodes(g, predict)
    except Exception:
        print("e1")
        return float('inf')

    kf = KalmanFilter(F, B, H, Q, R, P.copy(), x=x_est.copy())

    for x_true, z in traj:
        try:
            xp, P, y, S,  K, x_est  = execute(predict, [x_est, F, P, Q, z, R])

            if xp.shape != (dim,) or P.shape != (dim, dim):
                print("e2")
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isinf(xp)) or \
               np.any(np.isnan(P)) or np.any(np.isinf(P)):
                print("e3")
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                print("e4")
                return float('inf')

            # Raw prediction error
            err_pred = x_true - xp
            if np.any(np.isnan(err_pred)) or np.any(np.isinf(err_pred)):
                print("e5")
                return float('inf')
            pred_errors.append(np.dot(err_pred, err_pred))

            # Kalman-style update
            #K = (P @ H.T) @ np.linalg.inv(S)
            #x_est = xp + (K @ y)
            #P = (np.eye(dim) - (K @ H)) @ P

            kf.predict(u)
            kf.update(z)

            # Updated prediction error
            error = x_true - x_est
            if error.shape != (dim,) or np.any(np.isnan(error)) or np.any(np.isinf(error)):
                print("e6")
                return float('inf')

            squared_errors.append(np.dot(error, error))

        except Exception as e:
            print("e: ",e)
            print("predict : ",gp.as_graphviz(g, predict))
            print("e7")
            return float('inf')

    if not squared_errors or not pred_errors:
        print("e8")
        return float('inf')

    # Combine losses
    mse_updated = np.mean(squared_errors)
    mse_pred = np.mean(pred_errors)
    combined_loss = alpha * mse_updated + (1 - alpha) * mse_pred
    return combined_loss



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
            #P = (I - (K @ H)) @ P

            x_true = kf.predict(u)
            x_kalman = kf.update(z)

            if x_true.shape != xp.shape:
                return float('inf')

            diff_current = xx - x_kalman
            if np.any(np.isinf(diff_current)) or np.any(np.isnan(diff_current)):
                return float('inf')
            diff.append(diff_current @ diff_current.T)

        except (ValueError, TypeError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
            return float('inf')

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
g.n = 17
g.o = 6
g.a = 2
g.p = 0
g.lmb = 1000

kf  = gp.build(
        g,
        #  0     1    2    3   4    5    6        7        8           9        10       11     12    13   14        15       16    17       18      19  20    21   22   23   24
        ["i0", "i1","i2","i3","i4","i5","matmul","matmul", "transpose","matmul", "add","minus","add","inv","matmul","matmul","add","matmul","minus","o0","o1","o2","o3","o4","o5"],#
        [
            (1, 6), # x = (F @ xx)
            (0, 6),
            (1, 7), # (F @ P)
            (2, 7),
            (1, 8), # F.T
            (7, 9), #  ((F @ P) @ F.T)
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
            (14,15),  # xx = xp + (K @ y)
            (11,15),  
            (6,16),
            (15,16),
            (14,17),   #P = P - (K @ P)
            (10,17),
            (10,18),
            (17,18),            
            (6, 19),
            (18, 20),
            (11, 21),
            (12, 22),
            (14, 23),
            (16, 24),
        ],
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
        score = distance_from_kalman_filter(predict,trajectory)
        #score = distance_from_target_function(best_predict,trajectory)
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
sys.stdout.flush()
