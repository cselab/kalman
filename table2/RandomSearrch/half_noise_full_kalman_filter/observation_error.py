import numpy as np

# === System and noise model (matches your GP/Kalman setup) ===
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)


def generate_trajectory(length=500, seed=None):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    traj = []
    for _ in range(length):
        x = F @ x + cQ @ abs(rng.normal(0, 1, dim))
        z = H @ x + cR @ abs(rng.normal(0, 1, dim))
        traj.append((x.copy(), z.copy()))
    return traj


def compute_mse_per_trajectory(trajectories):
    mse_list = []
    for traj in trajectories:
        errors = [(np.linalg.norm(x_true - z)**2) for x_true, z in traj]
        mse_list.append(np.mean(errors))
    return np.array(mse_list)


# === Run the experiment ===
test_trajectories = [generate_trajectory(seed=32 + i) for i in range(50)]
mse_list = compute_mse_per_trajectory(test_trajectories)
mse = np.mean(mse_list)
stderr_real = np.std(mse_list, ddof=1) / np.sqrt(len(mse_list))

print(
    f"📉 MSE of observations vs. true state (test set): {mse:.6f} ± {stderr_real:.6f}\n"
)
