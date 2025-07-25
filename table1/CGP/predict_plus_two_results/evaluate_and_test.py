import os
import re
import numpy as np
import math
import sys
# === Define 'g' and GP operators ===
class g:
    pass

def execute(gen, x):
    return gp.execute(g, gen, x)

def minus(inp, args):
    return inp[0] - inp[1]

def matmul(inp, args):
    return inp[0] @ inp[1]

def add(inp, args):
    return inp[0] + inp[1]

def transpose(inp, args):
    return inp[0].T

g.nodes = (matmul, minus, add, transpose)
g.names = ("matmul", "minus", "add", "transpose")
g.arity = (2, 2, 2, 1)
g.args = (0, 0, 0, 0)
g.i = 6
g.n = 9
g.o = 4
g.a = 2
g.p = 0
g.lmb = 1000

import gp  # This assumes gp module with as_graphviz exists

log_dir = './'
digraphs = []
gen_island_pattern = re.compile(r'Generation\s+(\d+)\s+-\s+Island\s+(\d+).*?BEST Graph\s+:\s+(digraph \{.*?\n\})', re.DOTALL)

def from_graphviz(g, dot_str):
    edges = []
    node_labels = {}
    output_targets = {}
    output_node_ids = {}

    for line in dot_str.splitlines():
        line = line.strip()
        if line.startswith("digraph") or line == "{" or line == "}":
            continue
        label_match = re.match(r'(\d+) \[label = (.+)\]', line)
        edge_match = re.match(r'(\d+) -> (\d+)(?: \[color = (.+)\])?', line)
        if label_match:
            node = int(label_match[1])
            label = label_match[2].strip().strip('"')
            node_labels[node] = label
        elif edge_match:
            src = int(edge_match[1])
            tgt = int(edge_match[2])
            edges.append((src, tgt))

    width = 1 + g.a + g.p
    total_rows = g.i + g.n + g.o
    gen = np.zeros((total_rows, width), dtype=int)

    input_map = {}
    for src, tgt in edges:
        input_map.setdefault(tgt, []).append(src)

    func_nodes = [
        node_id for node_id, label in node_labels.items()
        if not label.startswith("i") and not label.startswith("o")
    ]
    func_nodes = sorted(func_nodes)

    node_id_to_row = {}
    row_to_graphviz_id = {}

    for i, node_id in enumerate(func_nodes[:g.n]):
        row = g.i + i
        node_id_to_row[node_id] = row
        row_to_graphviz_id[row] = node_id
        op_name = node_labels[node_id].split(',')[0].strip('"')
        op_idx = g.names.index(op_name)
        gen[row, 0] = op_idx
        for j, src in enumerate(input_map.get(node_id, [])):
            if j < g.a:
                gen[row, 1 + j] = node_id_to_row.get(src, src)

    for i in range(len(func_nodes), g.n):
        row = g.i + i
        gen[row, 0] = g.names.index("add")
        gen[row, 1] = 0
        gen[row, 2] = 1
        row_to_graphviz_id[row] = 1000 + i  # assign dummy ID

    for o_idx in range(g.o):
        gv_output_node_id = 15 + o_idx
        for node_id, label in node_labels.items():
            if label == f"o{o_idx}":
                srcs = input_map.get(node_id, [])
                if srcs:
                    src_id = node_id_to_row.get(srcs[0], srcs[0])
                    gen[g.i + g.n + o_idx, 1] = src_id
                    output_targets[o_idx] = src_id
                    output_node_ids[o_idx] = gv_output_node_id
    print(output_node_ids)

    return gen, output_node_ids, row_to_graphviz_id


def normalize_dot(dot):
    lines = dot.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip() not in {"digraph {", "}"}]
    return sorted(lines)
def normalize_dot_structure(dot_str):
    edges = []
    labels = {}

    for line in dot_str.strip().splitlines():
        line = line.strip()
        node_match = re.match(r'^(\d+)\s+\[label = (.+?)\]$', line)
        edge_match = re.match(r'^(\d+)\s+->\s+(\d+)', line)

        if node_match:
            node, label = int(node_match[1]), node_match[2].strip()
            labels[label] = labels.get(label, 0) + 1  # Count labels (i.e., 2 i0s vs 2 o3s)

        elif edge_match:
            src, dst = int(edge_match[1]), int(edge_match[2])
            edges.append((src, dst))

    return {
        "edges": sorted(edges),
        "label_counts": dict(sorted(labels.items()))
    }

def patch_output_labels(dot_str, output_node_ids):
    #print("dot_str : ",dot_str)
    lines = dot_str.strip().splitlines()
    header = [""]
    body = []
    footer = ["}"]

    # Remove any output label lines
    for line in lines:
        if re.search(r'\[label = o\d+\]', line):
            break
        if re.match(r'\d+ -> \d+', line):
            body.append(line)
        elif line.strip() != "digraph {" and line.strip() != "}":
            body.append(line)
        #print("output_node_ids : ",output_node_ids)
    for o_idx, output_node_id in sorted(output_node_ids.items()):
        src_id = predict[output_node_id, 1]
        body.append(f"  {output_node_id} [label = o{o_idx}]")
        body.append(f"  {src_id} -> {output_node_id}")

    return "\n".join(header + body + footer)

for root, _, files in os.walk(log_dir):
    for file in files:
        if file.endswith('.out'):
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()
                matches = gen_island_pattern.findall(content)
                for gen_num, island, digraph in matches:
                    filename = f"gen{gen_num}_island{island}.gv"
                    #with open(filename, 'w') as out_file:
                    #    out_file.write(digraph)
                    try:
                        # Convert from .gv string to genotype
                        predict, output_node_ids, row_to_graphviz_id = from_graphviz(g, digraph)

                        # Generate Graphviz from genotype (has internal row numbers like 6, 7, 8...)
                        raw_dot = gp.as_graphviz(g, predict)

                        # Replace internal node IDs with original Graphviz node IDs
                        remapped_dot_lines = []
                        for line in raw_dot.strip().splitlines():
                            line = line.strip()
                            # Match node declarations
                            output_match = re.match(r'^(\d+)\s+\[label = o(.+?)\]$', line)
                            if output_match:
                                break
                            node_match = re.match(r'^(\d+)\s+\[label = (.+?)\]$', line)
                            if node_match:
                                node_id, label = int(node_match[1]), node_match[2]
                                new_id = row_to_graphviz_id.get(node_id, node_id)
                                remapped_dot_lines.append(f"  {new_id} [label = {label}]")
                                continue

                            # Match edges
                            edge_match = re.match(r'^(\d+)\s+->\s+(\d+)(.*)', line)
                            if edge_match:
                                src, tgt, rest = int(edge_match[1]), int(edge_match[2]), edge_match[3]
                                new_src = row_to_graphviz_id.get(src, src)
                                new_tgt = row_to_graphviz_id.get(tgt, tgt)
                                remapped_dot_lines.append(f"  {new_src} -> {new_tgt}{rest}")
                                continue

                            # Copy any unrecognized lines
                            remapped_dot_lines.append(f"  {line}")

                        # Append output nodes (forced to 15–18)
                        for o_idx in range(g.o):
                            output_node_id = 15 + o_idx
                            src_internal = predict[g.i + g.n + o_idx, 1]
                            src_graphviz = row_to_graphviz_id.get(src_internal, src_internal)
                            remapped_dot_lines.append(f"  {output_node_id} [label = o{o_idx}]")
                            remapped_dot_lines.append(f"  {src_graphviz} -> {output_node_id}")

                        regenerated = "\n".join(remapped_dot_lines) + "\n}"

                        # Final comparison
                        if normalize_dot(digraph) != normalize_dot(regenerated) and normalize_dot(patch_output_labels(digraph,output_node_ids)) != normalize_dot(patch_output_labels(regenerated,output_node_ids)):
                            print("digraph :", digraph)
                            print("regenerated:", regenerated)
                            print("patch_output_labels1:", patch_output_labels(digraph,output_node_ids))
                            print("patch_output_labels2:", patch_output_labels(regenerated,output_node_ids))
                            print(f"Mismatch in {filename} after roundtrip conversion.")
                            exit(1)
                        digraphs.append((filename, predict))

                    except Exception as e:
                        print(f"Failed to convert {filename}: {e}")

print(f"Extracted and parsed {len(digraphs)} digraphs.")


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

def distance_from_kalman_filter(predict, traj):
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
            out = execute(predict, [xx, F, P, Q, z, R])
            xp, P, y, S = out #[2:]
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
            K = (P @ H.T) @ np.linalg.inv(S)
            xx = xp + (K @ y)
            I = np.eye(dim)
            P = (I - (K @ H)) @ P

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
            out = execute(predict, [x_est, F, P, Q, z, R])
            #print("len out : ",len(out))
            xp, P, y, S = out #[2:]

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
            K = (P @ H.T) @ np.linalg.inv(S)
            x_est = xp + (K @ y)
            P = (np.eye(dim) - (K @ H)) @ P

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




g.nodes = (matmul, minus, add, transpose)
g.names = ("matmul","minus","add","transpose")
g.arity = (2,2,2,1)
g.args = (0,0,0,0)

g.i = 6
g.n = 9
g.o = 4
g.a = 2
g.p = 0
g.lmb = 1000

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
    traj = generate_trajectory(length=500, seed=12345 + trial)
    validation_trajectories.append(traj)




best_score = float('inf')
best_predict = None
seen_hashes = set()
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

test_trajectories = []
all_scores = []
for trial in range(50):
    traj = generate_trajectory(length=500, seed=32 + trial)
    test_trajectories.append(traj)
if best_predict is not None:
    sys.stdout.write(f"\n🏆 Best Graph Score: {best_score}"+"\n")
    g.n = 7
    sys.stdout.write(f"🏁 Best Graphviz 7:\n{gp.as_graphviz(g, best_predict)}"+"\n")
    g.n = 9
    sys.stdout.write(f"🏁 Best Graphviz 9:\n{gp.as_graphviz(g, best_predict)}"+"\n")

    sys.stdout.write(f"🏁 Best Graphviz:\n{gp.as_graphviz(g, best_predict)}"+"\n")
    g.n = 9
    all_scores =[]
    for trajectory in test_trajectories:
        #score = distance_from_kalman_filter(predict,trajectory)
        score = distance_from_target_function(best_predict,trajectory)
        all_scores.append(score)
    score = np.mean(all_scores)

    #score = distance_from_kalman_filter(predict)
    sys.stdout.write(f"Distance Score of Kalman Filter: {score:.6f}"+"\n")
    sys.stdout.flush()