import numpy as np
import sys

# Kalman-like function to evaluate
def approximate(x, F, P, Q, z, R):
    xp = F @ x
    P = F @ P @ F.T + Q
    y = z - xp
    S = P + R
    K = P @ np.linalg.inv(S)
    x = xp + K @ y
    P = (np.eye(F.shape[0]) - K) @ P
    return xp, P, y, S, K, x

# Parameters
dim = 2
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)

# Trajectory generation function
def generate_trajectory(length=500, seed=None):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    traj = []
    for _ in range(length):
        x = F @ x + cQ @ np.abs(rng.normal(0, 1, dim))
        z = H @ x + cR @ np.abs(rng.normal(0, 1, dim))
        traj.append((x.copy(), z.copy()))
    return traj

# Generate test trajectories
test_trajectories = []
for trial in range(50):
    traj = generate_trajectory(length=500, seed=32 + trial)
    test_trajectories.append(traj)

# Evaluate approximate() on test set
mse_approx_trajectories = []

for trajectory in test_trajectories:
    x_est = np.array([0.0, 0.0])
    P = np.eye(dim)
    squared_errors = []

    for x_true, z in trajectory:
        try:
            xp, P, y, S, K, x_est = approximate(x_est, F, P, Q, z, R)

            err = x_true - x_est
            if np.any(np.isnan(err)) or np.any(np.isinf(err)):
                raise ValueError("Invalid error value in trajectory.")
            squared_errors.append(np.dot(err, err))

        except Exception as e:
            squared_errors = [float('inf')]
            break

    mse = np.mean(squared_errors) if squared_errors else float('inf')
    mse_approx_trajectories.append(mse)

# Final output
sys.stdout.write(f"MSE real trajectories for APPROXIMATE on test set : {np.mean(mse_approx_trajectories):.6f}\n")
sys.stdout.flush()
