import scipy.linalg
import utils
import sys
import numpy as np
import random
import subprocess
import statistics
import numpy as np
from numpy.linalg import LinAlgError, inv
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
class g:
    pass

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F.copy()  # State transition matrix
        self.B = B.copy()  # Control input matrix
        self.H = H.copy()  # Observation matrix
        self.Q = Q.copy()  # Process noise covariance
        self.R = R.copy()  # Measurement noise covariance
        self.P = P.copy()  # Estimate error covariance
        self.x = x.copy()  # Initial state estimate

    def predict(self, u=np.zeros((1,1))):
        """ Predict next state """
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        """ Correct state estimate with a measurement """
        self.y = z - (self.H @ self.x)  # Measurement residual
        self.S = (self.H @ (self.P @ self.H.T)) + self.R  # Residual covariance
        self.K = ((self.P @ self.H.T) @ np.linalg.inv(self.S))  # Kalman gain
        self.x = self.x + (self.K @ self.y)
        I = np.eye(self.F.shape[0])
        self.P = ((I - (self.K @ self.H)) @ self.P)
        return self.x

def create_function_from_string(function_code):
    exec(function_code, globals())  # Executes and defines `dynamic_function`

    # Now, `dynamic_function` is available
    return fun  # Return the created function






def evaluate(predict):
    np.random.seed(42)
    
    dim = 2
    dt = 0.1
    
    # Initial states
    x = np.array([0, 0], dtype=float)
    u = np.array([0, 0], dtype=float)
    
    # System and measurement matrices
    F = np.array([[1, dt],
                  [0, 1]], dtype=float)
    B = np.eye(dim)
    H = np.eye(dim)
    
    # Process and measurement noise
    Q = np.array([[0.25 * dt**4, 0.5 * dt**3],
                  [0.5 * dt**3,      dt**2]], dtype=float)
    Q += 1e-7 * np.eye(dim)
    
    R = 0.1 * np.eye(dim)
    R += 1e-7 * np.eye(dim)
    
    # State for the manual update portion
    xx = np.array([0, 0], dtype=float)
    P = np.eye(dim)
    
    # Kalman filter instance
    kf = KalmanFilter(F, B, H, Q, R, P, x)
    
    # Prepare for the loop
    trace = []
    mse_true_trajectory = 0.0
    
    # Attempt to parse the user-defined function
    try:
        new_function = create_function_from_string(predict)
    except Exception:
        return np.inf
    
    # Simulation loop
    for _ in range(200):
        try:
            # Simulate the true system
            x = F @ x + B @ u + np.linalg.cholesky(Q) @ np.random.normal(0, 3, dim)
            z = H @ x + np.linalg.cholesky(R) @ np.random.normal(0, 4, dim)
            
            # Kalman filter prediction and update
            x_true = kf.predict(u)
            kf.update(z)
            
            # Manual update step using the user-defined function
            xp, P = new_function(xx, F, P, Q)
            y = z - (H @ xp)
            S = H @ (P @ H.T) + R
            K = (P @ H.T) @ np.linalg.inv(S)
            xx = xp + (K @ y)
            
            I = np.eye(dim)
            P = (I - (K @ H)) @ P
            
            # Check dimension consistency
            if x.shape != x_true.shape or x_true.shape != xp.shape:
                return np.inf
            
            trace.append((x, z, xx))

            # Accumulate MSE (based on the difference between x_true and xp)
            diff = x_true - xp
            if diff @ diff.T < 0:
                return np.inf
            mse_true_trajectory += (diff @ diff.T) / 1000.0
        except Exception:
            return np.inf
    return mse_true_trajectory



def example():
    p = 2
    q = 10
    x = [random.randint(-p, p)]
    for i in range(N - 1):
        x.append(x[-1] + random.randint(-p, p))
        p, q = q, p
    return np.array(x, dtype=dtype)


# Make everything global because we are going to need it in the functions 

dim = 2
dt = 0.1
x = np.array([0, 0], dtype=float)
u = np.array([0, 0], dtype=float)
F = np.array([[1, dt], [0, 1]], dtype=float)
B = np.eye(dim)
Q = np.array([[1 / 4 * dt**4, 1 / 2 * dt**3], [1 / 2 * dt**3, dt**2]],
            dtype=float)
R = 0.1 * np.eye(dim)
H = np.eye(dim)

xx = np.array([0, 0], dtype=float)
P = np.eye(dim)



##print(hasattr(gp, "build"))  # Should return True if 'build' exists
Hash = set()
dtype = float
random.seed(time.time())
N = 100
def matmul(inp1, inp2):
    return inp1 @ inp2
def add(inp1, inp2):
    return inp1 + inp2
def transpose(inp1):
    return inp1.T