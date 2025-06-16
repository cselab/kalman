import os
import glob
import re
import numpy as np
import math
import sys
import textwrap

# === Kalman Constants ===
dim = 2
u = np.zeros(dim)
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
I = np.eye(dim)
B = np.eye(dim)

# === Kalman Filter Reference ===
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
        self.x = (self.F @ self.x)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = self.P - K @ self.P
        return self.x

# === Trajectory Generator ===
def generate_trajectory(N=500, seed=123):
    x = np.zeros(dim)
    traj = []
    rng = np.random.default_rng(seed)
    for _ in range(N):
        x = F @ x + cQ @ rng.normal(0, 1, dim)
        z = H @ x + cR @ rng.normal(0, 1, dim)
        traj.append((x.copy(), z.copy()))
    return traj

# === Generate Validation & Test Sets ===
validation_trajectories = [generate_trajectory(500, seed=12 + i) for i in range(50)]
test_trajectories = [generate_trajectory(500, seed=32 + i) for i in range(50)]

# === Execute Candidate Code ===
def create_function_from_string(function_code):
    try:
        exec(function_code, globals())
        return fun
    except SyntaxError as e:
        print(f"\n❌ SyntaxError on line {e.lineno}: {e.msg}")
        lines = function_code.splitlines()
        for i, line in enumerate(lines, 1):
            marker = ">>>" if i == e.lineno else "   "
            print(f"{marker} {i:>3}: {line}")
        print("-" * 60)
        return None
    except Exception as e:
        print(f"\n❌ Exec error: {e}")
        print("Function with error:")
        print(function_code)
        print("-" * 60)
        return None

# === Evaluate Across Multiple Trajectories ===
def evaluate_graph(predict_code, traj_list):
    print("⚙️ Evaluating function...")
    predict_fn = create_function_from_string(predict_code)
    if not predict_fn:
        return float("inf")

    total_score = 0.0
    failed_runs = 0

    for traj_idx, traj in enumerate(traj_list):
        xx = np.zeros(dim)
        P = np.eye(dim)
        kf = KalmanFilter(F, B, H, Q, R, P.copy(), x=xx.copy())

        total_diff = 0.0

        for t, (x_true, z) in enumerate(traj):
            try:
                xp, P_pred, y, S = predict_fn(xx.copy(), F.copy(), P.copy(), Q.copy(), z.copy(), R.copy())

                if xp.shape != (dim,) or P_pred.shape != (dim, dim):
                    raise ValueError("Invalid shape")
                if np.any(np.isnan(xp)) or np.any(np.isnan(P_pred)) or np.any(np.isinf(xp)) or np.any(np.isinf(P_pred)):
                    raise ValueError("NaN/Inf detected")

                K = (P_pred @ H.T) @ np.linalg.inv(S)
                xx = xp + (K @ y)
                P = (I - (K @ H)) @ P_pred

                x_ref = kf.predict(u)
                kf.update(z)

                diff = x_true - xx
                diff_scalar = float(diff @ diff.T)
                total_diff += diff_scalar / len(traj)

            except Exception as e:
                print(f"❌ Error at traj {traj_idx}, step {t}: {e}")
                failed_runs += 1
                total_diff = float("inf")
                break

        total_score += total_diff

    if failed_runs == len(traj_list):
        return float("inf")

    return total_score / (len(traj_list) - failed_runs)

# === Parse .out Files ===
def load_functions_from_file(path):
    print(f"📄 Loading from: {path}")
    with open(path, "r") as f:
        text = f.read()

    pattern = re.compile(
        r'\[Process\s+\d+\]\s+best score so far:\s*([\d\.eE\+-]+),\s*content:\s*(def fun\(.*?)(?=\n\[Process|\Z)',
        re.DOTALL
    )
    matches = pattern.findall(text)
    valid_candidates = []

    for score, raw_code in matches:
        code = textwrap.dedent(raw_code.strip())
        code = re.sub(r'^def\s+fun\s*\(', 'def fun(', code)

        if "return" not in code or len(code.splitlines()) < 4:
            continue

        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            continue

        valid_candidates.append((float(score), code))

    return valid_candidates

# === Main Execution ===
def main():
    out_files = glob.glob("*.out")
    if not out_files:
        print("❌ No .out files found.")
        return

    best_score = float("inf")
    best_code = None

    for file in out_files:
        print(f"\n🚀 Processing file: {file}")
        candidates = load_functions_from_file(file)

        if not candidates:
            print("❌ No valid candidates found.")
            continue

        for i, (reported_score, code) in enumerate(candidates):
            print(f"\n🔬 Function {i + 1} (Reported score: {reported_score}):")
            actual_score = evaluate_graph(code, validation_trajectories)
            print(f"📉 Score: {actual_score}")

            if actual_score < best_score:
                best_score = actual_score
                best_code = code

        if best_code:
            print(f"\n🏆 Best Function in {file}:")
            print(best_code)
            print(f"\n✅ Best Validation Score: {best_score:.6f}")
            test_score = evaluate_graph(best_code, test_trajectories)
            print(f"🧪 Test Score: {test_score:.6f}")
        else:
            print("❌ No valid executable function succeeded.")

if __name__ == "__main__":
    main()
