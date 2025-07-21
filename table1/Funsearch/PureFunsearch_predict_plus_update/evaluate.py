import scipy.linalg
import utils
import sys
import numpy as np
import random
import subprocess
import statistics
from numpy.linalg import LinAlgError, inv
import pathlib
import matplotlib.pyplot as plt
import scipy
import time
import math

class g:
    pass

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
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y = z - (self.H @ self.x)
        self.S = (self.H @ (self.P @ self.H.T)) + self.R
        self.K = ((self.P @ self.H.T) @ np.linalg.inv(self.S))
        self.x = self.x + (self.K @ self.y)
        I = np.eye(self.F.shape[0])
        self.P = ((I - (self.K @ self.H)) @ self.P)
        return self.x

def create_function_from_string(function_code):
    exec(function_code, globals())  # expects a function named `fun`
    return fun

def evaluate_graph(predict):
    xx = np.array([0, 0], dtype=float)
    P = np.eye(dim)
    F = np.array([[1, 1], [0, 1]], dtype=float)
    B = np.eye(dim)
    cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
    Q = cQ @ cQ.T
    cR = np.eye(dim)
    R = cR @ cR.T
    H = np.eye(dim)
    diff = []

    try:
        new_function = create_function_from_string(predict)
    except Exception as e:
        return float('inf')

    for x, z in traj:
        try:
            result = new_function(xx, F, P, Q, z, R)
            if not isinstance(result, tuple) or len(result) != 6:
                return float('inf')

            xp, P, y, S, K, xx = result
            #P = P - K @ P
            if xx.shape != (dim,) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xx)) or np.any(np.isnan(P)) or \
               np.any(np.isinf(xx)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xx) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')

            if xx.shape != x.shape:
                return float('inf')

            diff_current = xx - x
            if np.any(np.isnan(diff_current)) or np.any(np.isinf(diff_current)):
                return float('inf')

            diff.append(diff_current @ diff_current.T)

        except Exception as e:
            return float('inf')

    if not diff:
        return float('inf')

    loss = np.mean(diff)
    return loss if not math.isnan(loss) else float('inf')

def example():
    p = 2
    q = 10
    x = [random.randint(-p, p)]
    for i in range(N - 1):
        x.append(x[-1] + random.randint(-p, p))
        p, q = q, p
    return np.array(x, dtype=dtype)

# Global setup
dim = 2
dt = 0.1
N = 100
dtype = float
random.seed(time.time())

# Model matrices
x = np.array([0, 0], dtype=float)
u = np.array([0, 0], dtype=float)
F = np.array([[1, 1], [0, 1]], dtype=float)
B = np.eye(dim)
cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
I = np.eye(dim)

# Trajectory generation
traj = []
nprng = np.random.default_rng(seed=12345)
for t in range(2000):
    x = F @ x + cQ @ nprng.normal(0, 1, dim)
    z = H @ x + cR @ nprng.normal(0, 1, dim)
    traj.append((x, z))

# Optional utility functions
def matmul(inp1, inp2):
    return inp1 @ inp2

def add(inp1, inp2):
    return inp1 + inp2

def transpose(inp1):
    return inp1.T
