import re
import numpy as np
import math
import sys

# === Constants & Matrices ===
dim = 2
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
u = np.zeros(dim)

# === Kalman Filter ===
class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x = x.copy()

    def predict(self, u=np.zeros((2,))):
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

def get_F_Q(effective_dt):
    F = np.array([[1, effective_dt], [0, 1]], dtype=float)
    G = np.array([[0.5 * effective_dt ** 2], [effective_dt]])
    Q = G @ G.T
    return F, Q

# === Trajectory Generator ===
def generate_trajectory(length=500, seed=42):
    dim = 2
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

# === Evaluation Functions ===
def distance_from_target_function(func, traj, alpha=1.0):
    x_est = np.zeros(dim)
    P = np.eye(dim)
    squared_errors = []
    pred_errors = []

    for x_true, z, F_dyn, Q_dyn in traj:
        try:
            xp, P_new, y, S, K, x_est_apx = func(x_est.copy(), F_dyn.copy(), P.copy(), Q_dyn.copy(), z.copy(), R.copy())
            if xp.shape != (dim,) or x_est_apx.shape != (dim,) or P_new.shape != (dim, dim):
                return float("inf")
            if any(np.any(np.isnan(m)) or np.any(np.isinf(m)) for m in [xp, x_est_apx, P_new]):
                return float("inf")

            pred_errors.append(np.sum((x_true - xp) ** 2))
            squared_errors.append(np.sum((x_true - x_est_apx) ** 2))

            P = P_new.copy()
            x_est = x_est_apx.copy()
        except Exception:
            return float("inf")

    if not pred_errors or not squared_errors:
        return float("inf")

    mse_pred = np.mean(pred_errors)
    mse_post = np.mean(squared_errors)
    return alpha * mse_post + (1 - alpha) * mse_pred

# === Text Parsing & Execution ===
def load_functions_from_file(path):
    with open(path, "r") as f:
        text = f.read()
    pattern = re.compile(
        r'\[Process\s+\d+\]\s+best score so far:\s*(\S+),\s*content:\s*(def aproximate\(.*?)(?=\n\[Process|\Z)',
        re.DOTALL
    )
    matches = pattern.findall(text)
    return [(float(s), code.strip()) for s, code in matches if math.isfinite(float(s))]

def safe_exec_function(code):
    try:
        scope = {}
        exec(code, {"np": np, "math": math}, scope)
        return scope.get("aproximate")
    except Exception as e:
        sys.stdout.write(f"Execution error: {e}\n")
        sys.stdout.flush()
        return None

# === Main Evaluation ===
def main():
    input_path = "pytorch_18877820.out"  # 🔁 Update with your file
    candidates = load_functions_from_file(input_path)

    if not candidates:
        sys.stdout.write("\u274c No valid functions found.\n")
        sys.stdout.flush()
        return

    validation_trajectories = [generate_trajectory(500, seed=12 + i) for i in range(50)]
    test_trajectories = [generate_trajectory(500, seed=32 + i) for i in range(50)]

    best_score = float("inf")
    best_func = None
    best_code = ""

    for i, (reported_score, code) in enumerate(candidates):
        func = safe_exec_function(code)
        if not func:
            continue
        errors = [distance_from_target_function(func, traj) for traj in validation_trajectories]
        score = np.mean(errors)
        stderr = np.std(errors, ddof=1) / np.sqrt(len(errors))  # manual stderr
        sys.stdout.write(f"🔬 Function {i + 1}: MSE = {score:.6f}, StdErr = {stderr:.6f}\n")
        sys.stdout.flush()
        if score < best_score:
            best_score = score
            best_func = func
            best_code = code

    if best_func:
        sys.stdout.write("\n🏆 Best Function Code:\n\n")
        sys.stdout.write(best_code + "\n")

        val_errors = [distance_from_target_function(best_func, traj) for traj in validation_trajectories]
        val_score = np.mean(val_errors)
        val_stderr = np.std(val_errors, ddof=1) / np.sqrt(len(val_errors))
        sys.stdout.write(f"\n✅ Validation MSE: {val_score:.6f}, StdErr = {val_stderr:.6f}\n")
        sys.stdout.flush()

        test_errors = [distance_from_target_function(best_func, traj) for traj in test_trajectories]
        test_score = np.mean(test_errors)
        test_stderr = np.std(test_errors, ddof=1) / np.sqrt(len(test_errors))
        sys.stdout.write(f"\n🧪 Test MSE (vs target): {test_score:.6f}, StdErr = {test_stderr:.6f}\n")

        baseline_errors = []
        for traj in test_trajectories:
            kf = KalmanFilter(F, B, H, Q, R, np.eye(dim), x=np.zeros(dim))
            for x_true, z in traj:
                kf.predict(u)
                x_kf = kf.update(z)
                baseline_errors.append(np.sum((x_true - x_kf)**2))

        baseline_mse = np.mean(baseline_errors)
        baseline_stderr = np.std(baseline_errors, ddof=1) / np.sqrt(len(baseline_errors))
        sys.stdout.write(f"\n🏑 Kalman Filter Baseline MSE (vs target): {baseline_mse:.6f}, StdErr = {baseline_stderr:.6f}\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("❌ No valid executable function found.\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
