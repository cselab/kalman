import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Simulation settings
dt = 1
T = 200
steps = int(T / dt)
dim = 2

# System matrices
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[1, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
x_est = np.array([0.0, 0.0], dtype=float)
P = np.eye(dim, dtype=float)

# Generate trajectory
traj = []
x = np.array([0, 0], dtype=float)
nprng = np.random.default_rng(seed=12345)
for t in range(steps):
    x = F @ x + cQ @ nprng.normal(0, 1, dim)
    z = H @ x + cR @ nprng.normal(0, 1, dim)
    traj.append((x.copy(), z.copy()))

# Kalman Filter class
class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x = x.copy()

    def predict(self, u=np.array([0, 0], dtype=float)):
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - (self.H @ self.x)
        S = (self.H @ (self.P @ self.H.T)) + self.R
        K = ((self.P @ self.H.T) @ np.linalg.inv(S))
        self.x = self.x + (K @ y)
        self.P = self.P - (K @ self.P)
        return self.x

# Initialize filter and storage
kf = KalmanFilter(F, B, H, Q, R, P, x_est.copy())
true_states = []
observations = []
filtered_states = []

for x, z in traj:
    kf.predict()
    x_est = kf.update(z)
    true_states.append(x.flatten())
    observations.append(z.flatten())
    filtered_states.append(x_est.flatten())

# Convert to arrays
true_states = np.array(true_states)
filtered_states = np.array(filtered_states)
observations = np.array(observations)
t_vals = np.arange(steps) * dt

# Create side-by-side plots
fig, (ax_pos, ax_vel) = plt.subplots(1, 2, figsize=(16, 6))

# Position plot with inset
ax_pos.plot(t_vals, true_states[:, 0], 'k-', label='Truth')
ax_pos.plot(t_vals, filtered_states[:, 0], 'g-', label='Kalman Filter')
ax_pos.scatter(t_vals, observations[:, 0], color='r', s=10, label='Observations')
ax_pos.set_xlabel(r'$t$', fontsize=14)
ax_pos.set_ylabel(r'$x$', fontsize=14)
ax_pos.set_title('Position vs Time', fontsize=16)
ax_pos.legend()
ax_pos.grid(True)

# Inset in bottom right of position plot
zoom_start, zoom_end = 50, 60
zoom_mask = (t_vals >= zoom_start) & (t_vals <= zoom_end)

ax_inset = inset_axes(ax_pos, width="30%", height="30%", loc='lower right', borderpad=2)
ax_inset.plot(t_vals[zoom_mask], true_states[zoom_mask, 0], 'k-')
ax_inset.plot(t_vals[zoom_mask], filtered_states[zoom_mask, 0], 'g-')
ax_inset.scatter(t_vals[zoom_mask], observations[zoom_mask, 0], color='r', s=15)
ax_inset.set_xlim(zoom_start, zoom_end)
ax_inset.set_title("Zoomed-in", fontsize=10)
ax_inset.grid(True)
mark_inset(ax_pos, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

# Velocity plot
ax_vel.plot(t_vals, true_states[:, 1], 'k-', label='Truth')
ax_vel.plot(t_vals, filtered_states[:, 1], 'g-', label='Kalman Filter')
ax_vel.scatter(t_vals, observations[:, 1], color='r', s=10, label='Observations')
ax_vel.set_xlabel(r'$t$', fontsize=14)
ax_vel.set_ylabel(r'$\dot{x}$', fontsize=14)
ax_vel.set_title('Velocity vs Time', fontsize=16)
ax_vel.legend()
ax_vel.grid(True)

# Final layout adjustments and save
plt.subplots_adjust(wspace=0.3, left=0.06, right=0.98, top=0.92, bottom=0.1)
plt.savefig('./kalman_filter_combined.png', dpi=300)
plt.show()
