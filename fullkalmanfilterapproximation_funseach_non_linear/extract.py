import re
import numpy as np
import math
import sys

# === Constants & Matrices ===
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
I = np.eye(dim)
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

    def predict(self, u=np.zeros((1,1))):
        self.x = np.array([
            0.05 * self.x[0]**3-2*self.x[0],
            0.1 * np.sin(self.x[1])
        ])
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

# === Trajectory Generator ===
def generate_trajectory(length=500, seed=None):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    traj = []
    for _ in range(length):
        x = np.array([
            0.05 * x[0]**3-2*x[0],
            0.1 * np.sin(x[1])
        ])
        x = F @ x + cQ @ rng.normal(0, 1, dim)
        z = H @ x + cR @ rng.normal(0, 1, dim)
        traj.append((x.copy(), z.copy()))
    return traj

# === Evaluation Functions ===
def distance_from_target_function(func, traj, alpha=1.0):
    x_est = np.zeros(dim)
    P = np.eye(dim)

    squared_errors = []
    pred_errors = []

    kf = KalmanFilter(F, B, H, Q, R, P.copy(), x=x_est.copy())

    for x_true, z in traj:
        try:
            xp, P_new, y, S, K, x_est_apx = func(x_est.copy(), F.copy(), P.copy(), Q.copy(), z.copy(), R.copy())

            if xp.shape != (dim,) or x_est_apx.shape != (dim,) or P_new.shape != (dim, dim):
                return float("inf")
            if any(np.any(np.isnan(m)) or np.any(np.isinf(m)) for m in [xp, x_est_apx, P_new]):
                return float("inf")

            pred_errors.append(np.sum((x_true - xp)**2))

            kf.predict(u)
            kf.update(z)

            squared_errors.append(np.sum((x_true - x_est_apx)**2))
            P = P_new.copy()
            x_est = x_est_apx.copy()
        except Exception:
            return float("inf")

    if not pred_errors or not squared_errors:
        return float("inf")

    mse_pred = np.mean(pred_errors)
    mse_post = np.mean(squared_errors)
    return alpha * mse_post + (1 - alpha) * mse_pred

def distance_from_kalman_filter(func, traj):
    x_est = np.zeros(dim)
    P = np.eye(dim)
    kf = KalmanFilter(F, B, H, Q, R, P.copy(), x=np.zeros(dim))

    diffs = []

    for _, z in traj:
        try:
            _, P_new, _, _, _, x_est_apx = func(x_est.copy(), F.copy(), P.copy(), Q.copy(), z.copy(), R.copy())

            x_kf = kf.predict(u)
            kf.update(z)

            if x_est_apx.shape != x_kf.shape or np.any(np.isnan(x_est_apx)) or np.any(np.isinf(x_est_apx)):
                return float("inf")

            diffs.append(np.sum((x_est_apx - x_kf)**2))
            P = P_new.copy()
            x_est = x_est_apx.copy()
        except Exception:
            return float("inf")

    return np.mean(diffs) if diffs else float("inf")

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
        exec(code, {"np": np}, scope)
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
        score = np.mean([distance_from_target_function(func, traj) for traj in validation_trajectories])
        sys.stdout.write(f"🔬 Function {i + 1}: MSE = {score:.6f}\n")
        sys.stdout.flush()
        if score < best_score:
            best_score = score
            best_func = func
            best_code = code

    if best_func:
        sys.stdout.write("\n🏆 Best Function Code:\n\n")
        sys.stdout.write(best_code + "\n")
        sys.stdout.write(f"\n✅ Validation MSE: {best_score:.6f}\n")
        sys.stdout.flush()

        test_score_target = np.mean([distance_from_target_function(best_func, traj) for traj in test_trajectories])

        sys.stdout.write(f"\n🧪 Test MSE (vs target): {test_score_target:.6f}\n")

        baseline_errors = []
        for traj in test_trajectories:
            kf = KalmanFilter(F, B, H, Q, R, np.eye(dim), x=np.zeros(dim))
            for x_true, z in traj:
                kf.predict(u)
                x_kf = kf.update(z)
                baseline_errors.append(np.sum((x_true - x_kf)**2))
        baseline_mse = np.mean(baseline_errors)
        sys.stdout.write(f"\n🏑 Kalman Filter Baseline MSE (vs target): {baseline_mse:.6f}\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("❌ No valid executable function found.\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
