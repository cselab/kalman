import scipy.linalg
import sys
import numpy as np
import random
import subprocess
import statistics
from numpy.linalg import LinAlgError, inv
import pathlib
import multiprocessing
import matplotlib.pyplot as plt
import scipy
import time
import math
import heapq


def minus(inp, args):
    return inp[0] - inp[1]

def matmul(inp, args):
    return inp[0] @ inp[1]

def add(inp, args):
    return inp[0] + inp[1]

def transpose(inp, args):
    return inp[0].T

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x = x.copy()

    def predict(self, u=np.zeros((1,1))):
        self.x = (self.F @ self.x) 
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y = z - self.x
        self.S = self.P + self.R
        self.K = self.P @ np.linalg.inv(self.S)
        self.x = self.x + (self.K @ self.y)
        self.P = ( self.P - self.K @ self.P)
        return self.x

def create_function_from_string(function_code):
    exec(function_code, globals())  # Executes and defines `dynamic_function`

    # Now, `dynamic_function` is available
    return fun  # Return the created function

def evaluate_graph(predict):
    xx = np.array([0, 0], dtype=float)
    mse_true_trajectory = 0.0
    P = np.eye(dim)
    diff = 0


    kf = KalmanFilter(F, B, H, Q, R, P, x=np.array([0, 0], dtype=float))

    try:
        new_function = create_function_from_string(predict)
    except Exception:
        return np.inf

    loss = 0
    for x, z in traj:
        try:

            xp, P, y, S = new_function(xx, F, P, Q, z, R)
            #xp, P, y = execute(predict, [xx, F, P, Q, z])
            if xp.shape != (dim,) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isnan(P)) or \
               np.any(np.isinf(xp)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')

            #S = H @ (P @ H.T) + R
            K = (P @ H.T) @ np.linalg.inv(S)
            xx = xp + (K @ y)
            I = np.eye(dim)
            P = (I - (K @ H)) @ P

            x_true = kf.predict(u)
            kf.update(z)

            if x_true.shape != xp.shape:
                return float('inf')

            diff_current = x_true - xp
            if np.any(np.isinf(diff_current)) or np.any(np.isnan(diff_current)):
                return float('inf')
            diff += diff_current @ diff_current.T / len(traj)

        except Exception :
            return float('inf')

    return  diff ## loss if not math.isnan(loss) else float('inf')






N = 100
dim = 2

dt = 0.1
x = np.array([0, 0], dtype=float)
u = np.array([0, 0], dtype=float)
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
I = np.eye(dim)
B = np.eye(dim)
traj = []
nprng = np.random.default_rng(seed=12345)

for t in range(200):
    x = F @ x + cQ @ nprng.normal(0, 1, dim)
    z = H @ x + cR @ nprng.normal(0, 1, dim)
    traj.append((x, z))

