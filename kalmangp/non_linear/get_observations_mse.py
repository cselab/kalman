import numpy as np
import matplotlib.pyplot as plt
import gp  # Your custom GP module

# === GP Graph Setup ===
class g:
    pass

def minus(inp, args): return inp[0] - inp[1]
def matmul(inp, args): return inp[0] @ inp[1]
def add(inp, args): return inp[0] + inp[1]
def transpose(inp, args): return inp[0].T
def inv(inp, args): return np.linalg.inv(inp[0])

g.nodes = (matmul, minus, add, transpose, inv)
g.names = ("matmul", "minus", "add", "transpose", "inv")
g.arity = (2, 2, 2, 1, 1)
g.args = (0, 0, 0, 0, 0)
g.i = 6
g.n = 19
g.o = 6
g.a = 2
g.p = 0
g.lmb = 1000

kalman_filter = gp.build(
    g,
    ["i0", "i1", "i2", "i3", "i4", "i5", "matmul", "matmul", "transpose", "matmul",
     "add", "minus", "add", "inv", "matmul", "matmul", "add", "matmul", "minus", "o0",
     "o1", "o2", "o3", "o4", "o5"],
    [(1, 6), (0, 6), (1, 7), (2, 7), (1, 8), (7, 9), (8, 9), (9, 10), (3, 10),
     (4, 11), (6, 11), (10, 12), (5, 12), (12, 13), (10, 14), (13, 14), (14, 15),
     (11, 15), (6, 16), (15, 16), (14, 17), (10, 17), (10, 18), (17, 18), (6, 19),
     (18, 20), (11, 21), (12, 22), (14, 23), (16, 24)],
    []
)

def aproximate(x, F, P, Q, z, R):
    return gp.execute(g, kalman_filter, [x, F, P, Q, z, R])

# === Trajectory Generator with Delay ===
dim = 2
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)

def get_F_Q(effective_dt):
    F = np.array([[1, effective_dt], [0, 1]], dtype=float)
    G = np.array([[0.5 * effective_dt ** 2], [effective_dt]])
    Q = G @ G.T
    return F, Q

def generate_trajectory(length=100, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
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

# === Run and Evaluate One Trajectory ===
traj = generate_trajectory(seed=100)
x_est = np.zeros(dim)
P = np.eye(dim)

true_states = []
observations = []
predicted_states = []

for x_true, z, F_dyn, Q_dyn in traj:
    try:
        xp, P_new, y, S, K, x_est_apx = aproximate(x_est.copy(), F_dyn.copy(), P.copy(), Q_dyn.copy(), z.copy(), R.copy())
        predicted_states.append(xp)
        x_est = x_est_apx.copy()
        P = P_new.copy()
    except Exception as e:
        print("Graph execution failed:", e)
        predicted_states.append(np.full(dim, np.nan))

    true_states.append(x_true)
    observations.append(z)

true_states = np.array(true_states)
observations = np.array(observations)
predicted_states = np.array(predicted_states)

# === Plotting: Phase Plot (x₀ vs x₁ for first 10 steps) ===
plt.figure(figsize=(8, 6))

steps = slice(0, 10)
plt.plot(true_states[steps, 0], true_states[steps, 1], 'o-', label="True Trajectory", linewidth=2)
plt.plot(observations[steps, 0], observations[steps, 1], 'x', label="Observed", alpha=0.7)
plt.plot(predicted_states[steps, 0], predicted_states[steps, 1], 's-.', label="Predicted (Graph)", linewidth=2)

plt.title("Position vs Velocity (First 10 Steps)")
plt.xlabel("Position (x₀)")
plt.ylabel("Velocity (x₁)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("phase_plot_first10.png")
plt.show()
