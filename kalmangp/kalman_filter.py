import matplotlib.pyplot as plt
import numpy as np
import scipy



class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P, x):
        self.F = F  # State transition matrix
        self.B = B  # Control input matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Estimate error covariance
        self.x = x  # Initial state estimate

    def predict(self, u=np.zeros((1,1))):
        """ Predict next state """
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q

    def update(self, z):
        """ Correct state estimate with a measurement """
        y = z - (self.H @ self.x)  # Measurement residual
        S = (self.H @ (self.P @ self.H.T)) + self.R  # Residual covariance
        K = ((self.P @ self.H.T) @ np.linalg.inv(S))  # Kalman gain

        self.x = self.x + (K @ y)
        I = np.eye(self.F.shape[0])
        self.P = ((I - (K @ self.H) )@ self.P)
        return  self.x


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
Trace = []

kf = KalmanFilter(F, B, H, Q, R, P, x)

mse = 0
mse_data = 0
for t in range(100):
    x = F @ x + B @ u + scipy.linalg.sqrtm(Q) @ np.random.normal(0, 3, dim)
    z = H @ x + scipy.linalg.sqrtm(R) @ np.random.normal(0, 4, dim)

    xp = kf.predict(u)
    xx = kf.update(z)
    print(xp,xx)
    print((x - xx)**2/100)
    Trace.append((x, z, xx))
    mse += (x - xx)**2/100
    mse_data += (x - z)**2/100 
x, z, xx = zip(*Trace)

x0, x1 = zip(*x)
z0, z1 = zip(*z)
xx0, xx1 = zip(*xx)

# position
plt.plot(x0, "-b", z0, "o", xx0, "-r")
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki2.0.svg")
plt.close()

print(mse)
print(mse_data)
# velocity
plt.plot(x1, "-b", z1, "o", xx1, "-r")
plt.legend(["state", "observation", "estimate"])
plt.savefig("wiki2.1.svg")