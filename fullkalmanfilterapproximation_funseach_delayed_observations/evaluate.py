import numpy as np
import math
import re
import sys

# === Configuration ===
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
u = np.zeros(dim)

# === Kalman Filter Class ===
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
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x

# === Trajectory Generator ===
def get_F_Q(effective_dt):
    F = np.array([[1, effective_dt], [0, 1]], dtype=float)
    G = np.array([[0.5 * effective_dt ** 2], [effective_dt]])
    Q = G @ G.T
    return F, Q

def generate_trajectory(length=500, seed=42):
    cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
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

# === Evaluation Helpers ===
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

def evaluate_kalman_filter(trajectories):
    total_errors = []
    for traj in trajectories:
        x_est = np.zeros(dim)
        P_est = np.eye(dim)
        for x_true, z, F_dyn, Q_dyn in traj:
            kf = KalmanFilter(F_dyn, B, H, Q_dyn, R, P_est.copy(), x_est.copy())
            kf.predict(u)
            x_pred = kf.update(z)
            error = np.sum((x_true - x_pred) ** 2)
            total_errors.append(error)
    return np.mean(total_errors)

# === Function Crawling Helpers ===
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
        print(f"❌ Execution error: {e}")
        return None

# === Main ===
def main():
    # === Generate Datasets ===
    validation_trajectories = [generate_trajectory(seed=12 + i) for i in range(50)]
    test_trajectories = [generate_trajectory(seed=32 + i) for i in range(50)]

    # === PART 1: Kalman Filter Evaluation ===
    val_score_kf = evaluate_kalman_filter(validation_trajectories)
    test_score_kf = evaluate_kalman_filter(test_trajectories)
    print(f"✅ Kalman Filter Validation MSE: {val_score_kf:.6f}")
    print(f"🧪 Kalman Filter Test MSE:       {test_score_kf:.6f}")

    # === PART 2: Manual aproximate() Function ===
    my_function = '''
def aproximate(x, F, P, Q, z, R):
    xp = F @ x
    P = F @ P @ F.T + Q
    y = z - xp
    S = P + R
    K = P @ np.linalg.inv(S)
    x = xp + K @ y
    P = (np.eye(F.shape[0]) - K) @ P
    return xp, P, y, S, K, x
    '''
    scope = {}
    exec(my_function, {"np": np}, scope)
    aprox = scope["aproximate"]

    val_score_apx = np.mean([distance_from_target_function(aprox, traj) for traj in validation_trajectories])
    test_score_apx = np.mean([distance_from_target_function(aprox, traj) for traj in test_trajectories])
    print(f"\n🧪 Aproximate Function Evaluation:")
    print(f"✅ Validation MSE: {val_score_apx:.6f}")
    print(f"🧪 Test MSE:       {test_score_apx:.6f}")

    # === PART 3: Crawl from file ===
    print("\n🕷️ Crawling candidate functions from log file...")
    input_path = "pytorch_18588979.out"  # ✅ Change to your file if needed
    candidates = load_functions_from_file(input_path)

    best_score = float("inf")
    best_func = None
    best_code = ""

    for i, (reported_score, code) in enumerate(candidates):
        func = safe_exec_function(code)
        if not func:
            continue
        try:
            score = np.mean([distance_from_target_function(func, traj) for traj in validation_trajectories])
            print(f"🔬 Candidate {i + 1}: reported={reported_score:.6f}, val_MSE={score:.6f}")
            if score < best_score:
                best_score = score
                best_func = func
                best_code = code
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")

    if best_func:
        print("\n🏆 Best Crawled Function:\n")
        print(best_code)
        print(f"\n✅ Best Validation MSE: {best_score:.6f}")
        test_score = np.mean([distance_from_target_function(best_func, traj) for traj in test_trajectories])
        print(f"🧪 Best Test MSE:       {test_score:.6f}")
    else:
        print("❌ No valid candidate functions were found.")

if __name__ == "__main__":
    main()
