import os
import re
import math
import numpy as np
from textwrap import dedent

# === Kalman Filter Setup ===
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
R = np.eye(dim)
H = np.eye(dim)
B = np.eye(dim)
u = np.zeros(dim)


class KalmanFilter:

    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x = x.copy()

    def predict(self, u=np.zeros((1, 1))):
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


# === Synthetic Trajectory Generator ===
def generate_trajectory(length=500, seed=None):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    traj = []
    for _ in range(length):
        x = F @ x + cQ @ rng.normal(0, 1, dim)
        z = H @ x + R @ rng.normal(0, 1, dim)
        traj.append((x.copy(), z.copy()))
    return traj


# === Loss Function for Code-Based Predictors ===
def distance_from_target_function_code(predict_fn, traj):
    xx = np.array([0, 0], dtype=float)
    P = np.eye(dim)
    diff = []

    try:
        kf = KalmanFilter(F, B, H, Q, R, P=np.eye(dim), x=np.zeros(dim))
        for x, z in traj:
            try:
                xp, P_new, y, S = predict_fn(xx, F, P, Q, z, R)
                if xp.shape != (dim, ) or P_new.shape != (dim, dim):
                    return float('inf')
                if np.any(np.isnan(xp)) or np.any(np.isnan(P_new)) or \
                   np.any(np.isinf(xp)) or np.any(np.isinf(P_new)):
                    return float('inf')
                if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P_new) > 1e6:
                    return float('inf')

                #y = z - (H @ xp)
                #S = H @ (P_new @ H.T) + R
                K = (P_new @ H.T) @ np.linalg.inv(S)
                xx = xp + (K @ y)
                I = np.eye(dim)
                P = (I - (K @ H)) @ P_new

                x_true = kf.predict(u)
                kf.update(z)

                diff_current = x - xx
                if np.any(np.isinf(diff_current)) or np.any(
                        np.isnan(diff_current)):
                    return float('inf')

                diff.append(diff_current @ diff_current.T)
            except Exception:
                return float('inf')

        loss = np.mean(diff)
        return loss if not math.isnan(loss) else float('inf')
    except Exception:
        return float('inf')


# === Kalman Baseline ===
def kalman_baseline(traj):
    kf = KalmanFilter(F, B, H, Q, R, np.eye(dim), x=np.zeros(dim))
    errors = []
    for x, z in traj:
        kf.predict(u)
        x_est = kf.update(z)
        errors.append((x - x_est) @ (x - x_est))
    return np.mean(errors)


# === Function Extractor ===
def extract_functions_from_log(text):
    pattern = re.compile(r"content:\s*\n(def fun\(.*?\):(?:\n    .*)+)",
                         re.MULTILINE)
    return pattern.findall(text)


# === Evaluation Pipeline ===
def evaluate_all_functions_in_logs(base_dir="./"):
    validation_trajectories = [
        generate_trajectory(seed=100 + i) for i in range(30)
    ]
    best_score = float('inf')
    best_func_code = None
    best_fn = None
    func_counter = 0

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".out"):
                full_path = os.path.join(root, file)
                print(f"\n📂 Scanning file: {full_path}")
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    functions = extract_functions_from_log(content)

                    for idx, full_code in enumerate(functions):
                        func_counter += 1
                        try:
                            cleaned = dedent(full_code)
                            local_scope = {}
                            exec(cleaned, globals(), local_scope)
                            predict_fn = local_scope['fun']

                            scores = [
                                distance_from_target_function_code(
                                    predict_fn, traj)
                                for traj in validation_trajectories
                            ]
                            score = np.mean(scores)
                            print(f"  [#{func_counter:04}] Score: {score:.6f}")

                            if score < best_score:
                                best_score = score
                                best_func_code = cleaned
                                best_fn = predict_fn

                        except Exception as e:
                            print(
                                f"  [#{func_counter:04}] ❌ Error evaluating function: {e}"
                            )

                except Exception as e:
                    print(f"❌ Failed to read file {full_path}: {e}")

    print("\n🏆 Best Score on Validation Set:", best_score)
    print("📈 Best Function:\n")
    print(best_func_code or "No valid function found.")

    # === Final Evaluation ===
    if best_fn:
        test_trajectories = [
            generate_trajectory(seed=32 + i) for i in range(50)
        ]
        pred_scores = [
            distance_from_target_function_code(best_fn, traj)
            for traj in test_trajectories
        ]
        kalman_scores = [kalman_baseline(traj) for traj in test_trajectories]

        mean_pred = np.mean(pred_scores)
        stderr_pred = np.std(pred_scores, ddof=1) / np.sqrt(len(pred_scores))

        mean_kf = np.mean(kalman_scores)
        stderr_kf = np.std(kalman_scores, ddof=1) / np.sqrt(len(kalman_scores))

        print("\n📊 Final Test Performance")
        print(
            f"Evolved Predictor MSE     : {mean_pred:.6f} ± {stderr_pred:.6f}")


# === Main Entry ===
if __name__ == "__main__":
    evaluate_all_functions_in_logs(base_dir="./")  # Adjust folder if needed
