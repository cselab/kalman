import numpy as np

def get_F_Q(dt):
    F = np.array([[1, dt], [0, 1]])
    G = np.array([[0.5 * dt**2], [dt]])
    Q = G @ G.T
    return F, Q

def generate_trajectory(length=500, seed=0):
    dim = 2
    x = np.zeros(dim)
    cQ = np.array([[0.5, 0], [1, 0]])
    cR = np.eye(dim)
    R = cR @ cR.T
    H = np.eye(dim)
    rng = np.random.default_rng(seed)
    true_states = [x.copy()]
    traj = []

    for t in range(1, length):
        delay = rng.uniform(0.01, 0.3)
        dt_eff = 1.0 + delay
        F, Q = get_F_Q(dt_eff)
        x = F @ x + cQ @ rng.normal(0, 1, 2)
        true_states.append(x.copy())
        ε_idx = t - delay
        t0 = int(np.floor(ε_idx))
        t1 = min(t0 + 1, len(true_states) - 1)
        α = ε_idx - t0
        x_interp = (1 - α) * true_states[t0] + α * true_states[t1]
        z = H @ x_interp + cR @ rng.normal(0, 1, 2)
        traj.append((x.copy(), z.copy(), F, Q))

    return traj

def compute_observation_vs_truth_mse(trajectories):
    errors = []
    for traj in trajectories:
        for x_true, z, _, _ in traj:
            errors.append(np.linalg.norm(x_true - z) ** 2)
    return np.mean(errors)

# === Run on multiple test trajectories ===
test_trajectories = [generate_trajectory(seed=100 + i) for i in range(50)]
mse = compute_observation_vs_truth_mse(test_trajectories)
print(f"📉 MSE of observations vs. true trajectory: {mse:.6f}")
