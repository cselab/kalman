import numpy as np

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

    def predict(self, u=np.zeros(2)):
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

# === System parameters ===
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[1/2, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
u = np.zeros(dim)

# === Generate trajectories ===
def generate_trajectory(length=500, seed=None):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    traj = []
    for _ in range(length):
        x = F @ x + cQ @ rng.normal(0, 1, dim)
        z = H @ x + cR @ rng.normal(0, 1, dim)
        traj.append((x.copy(), z.copy()))
    return traj

# === Evaluate Kalman filter ===
def evaluate_kalman_filter(traj):
    x_est = np.zeros(dim)
    P = np.eye(dim)
    kf = KalmanFilter(F, B, H, Q, R, P.copy(), x_est.copy())
    squared_errors = []

    for x_true, z in traj:
        kf.predict(u)
        x_est = kf.update(z)
        error = x_true - x_est
        squared_errors.append(error @ error)

    return np.mean(squared_errors)

# === Main Evaluation on Test Set ===
test_trajectories = [generate_trajectory(length=500, seed=12 + i) for i in range(10)]
mse_scores = [evaluate_kalman_filter(traj) for traj in test_trajectories]

print("✅ Kalman Filter Evaluation on Test Set")
print(f"📈 Mean MSE over 10 trajectories: {np.mean(mse_scores):.6f}")
