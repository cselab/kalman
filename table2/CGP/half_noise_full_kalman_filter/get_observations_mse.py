import numpy as np

# System and noise model (matches your GP/Kalman setup)
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


def compute_observation_vs_truth_mse(trajectories):
    errors = []
    for traj in trajectories:
        for x_true, z in traj:
            errors.append(np.linalg.norm(x_true - z)**2)
    return np.mean(errors)


# === Run the experiment ===
test_trajectories = [generate_trajectory(seed=32 + i) for i in range(50)]
mse = compute_observation_vs_truth_mse(test_trajectories)
print(f"📉 MSE of observations vs. true state (test set): {mse:.6f}")
