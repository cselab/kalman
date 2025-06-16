import gp
import numpy as np
import os
import random
import sys
import pathlib


class g:
    pass


def seen(Hash, a):
    a = a.tobytes()
    ans = (a) in Hash
    if not ans:
        Hash.add((a))
    return ans


class KalmanFilter:

    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()  # State transition matrix
        self.B = B.copy()  # Control input matrix
        self.H = H.copy()  # Observation matrix
        self.Q = Q.copy()  # Process noise covariance
        self.R = R.copy()  # Measurement noise covariance
        self.P = P.copy()  # Estimate error covariance
        self.x = x.copy()  # Initial state estimate

    def predict(self, u=np.zeros((1, 1))):
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y = z - (self.H @ self.x)  # Measurement residual
        self.S = (self.H @ (self.P @ self.H.T)) + self.R  # Residual covariance
        self.K = ((self.P @ self.H.T) @ np.linalg.inv(self.S))  # Kalman gain
        self.x = self.x + (self.K @ self.y)
        I = np.eye(self.F.shape[0])
        self.P = ((I - (self.K @ self.H)) @ self.P)
        return self.x


def execute(gen, x):
    """Executes the gene 'gen' via gp.execute."""
    return gp.execute(g, gen, x)


def fun(predict):
    """Runs the main loop for the Kalman filter, comparing manual vs. GP-coded prediction."""
    np.random.seed(42)

    dim = 2
    dt = 0.1

    # Initial states
    x = np.array([0, 0], dtype=float)
    u = np.array([0, 0], dtype=float)

    # System and measurement matrices
    F = np.array([[1, dt], [0, 1]], dtype=float)
    B = np.eye(dim)
    H = np.eye(dim)

    # Process and measurement noise
    Q = np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
                 dtype=float)
    Q += 1e-7 * np.eye(dim)

    R = 0.1 * np.eye(dim)
    R += 1e-7 * np.eye(dim)

    # State for the manual update portion
    xx = np.array([0, 0], dtype=float)
    P = np.eye(dim)

    # Kalman filter instance
    kf = KalmanFilter(F, B, H, Q, R, P, x)

    mse_true_trajectory = 0.0

    cQ = np.linalg.cholesky(Q)
    cR = np.linalg.cholesky(R)
    I = np.eye(dim)
    try:
        for i in range(2000):
            x = F @ x + B @ u + cQ @ np.random.normal(0, 3, dim)
            z = H @ x + cR @ np.random.normal(0, 4, dim)
            x_true = kf.predict(u)
            kf.update(z)
            xp, P = execute(predict, [xx, F, P, Q])
            y = z - (H @ xp)
            S = H @ (P @ H.T) + R
            K = (P @ H.T) @ np.linalg.inv(S)
            xx = xp + (K @ y)

            P = (I - (K @ H)) @ P
            diff = x_true - xp
            if diff @ diff.T < 0:
                return np.inf
            mse_true_trajectory += (diff @ diff.T) / 1000.0
            if i == 5 and mse_true_trajectory > 0:
                return mse_true_trajectory
    except Exception as e:
        return np.inf
    return mse_true_trajectory


def matmul(inp, args):
    return inp[0] @ inp[1]


def add(inp, args):
    return inp[0] + inp[1]


def transpose(inp, args):
    return inp[0].T


g.nodes = matmul, add, transpose
g.names = "matmul", "add", "transpose"
g.arity = 2, 2, 1
g.args = 0, 0, 0
g.i = 4
g.n = 5
g.o = 2
g.a = max(g.arity)
g.p = 0
# Build a default gene as an example
predict0 = gp.build(
    g,
    [
        "i0", "i1", "i2", "i3", "matmul", "matmul", "transpose", "matmul",
        "add", "o0", "o1"
    ],
    [
        (1, 4),  # (F @ xx)
        (0, 4),
        (1, 5),  # (F @ P)
        (2, 5),
        (1, 6),  # F.T
        (5, 7),  # ((F @ P) @ F.T)
        (6, 7),
        (7, 8),  # ((F @ P) @ F.T) + Q
        (3, 8),
        (4, 9),
        (8, 10)
    ],
    [])
print("Initial predict0 cost:", fun(predict0))

best_score = np.inf
jid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
random.seed(jid)
dirname = f"{jid:08d}"
os.makedirs(dirname, exist_ok=True)
i = 0
cnt = 0
with open(os.path.join(dirname, "score"), "w") as score_file:
    while True:
        gene = gp.rand(g)
        score = fun(gene)
        cnt += 1
        if score < best_score:
            best_score = score
            score_file.write(f"{score:.16e} {cnt:010d}\n")
            score_file.flush()
            pathlib.Path(os.path.join(dirname, f"{i:08d}.gv")).write_text(
                gp.as_graphviz(g, gene))
            sys.stderr.write(f"Kalman.nodes.py: {score:.16e} {cnt:010d}\n")
            i += 1
        if cnt % 10000 == 0:
            sys.stderr.write(f"Kalman.nodes.py: {cnt:010d}\n")
