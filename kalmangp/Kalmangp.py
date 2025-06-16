import scipy.linalg
import gp
import sys
import numpy as np
import random
import subprocess
import statistics
import numpy as np
from numpy.linalg import LinAlgError, inv
import pathlib
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
class g:
    pass
def seen(a, b):
    a = a.tobytes()
    b = b.tobytes()
    ans = (a, b) in Hash
    if not ans:
        Hash.add((a, b))
    return ans

def seen(a):
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






def fun(predict):
    ###print("fun : ",predict)
    global dim, dt, x, F, B, Q, R, H, xx, P, u
    dim = 2
    dt = 0.1
    x = np.array([0, 0], dtype=float)
    u = np.array([0, 0], dtype=float)
    F = np.array([[1, dt], [0, 1]], dtype=float)
    B = np.eye(dim)
    R = 0.1 * np.eye(dim)
    Q = np.array([[1 / 4 * dt**4, 1 / 2 * dt**3], [1 / 2 * dt**3, dt**2]],
                dtype=float)
    Q = Q + 1e-7 * np.eye(Q.shape[0])
    R = R + 1e-7 * np.eye(R.shape[0])

    R = 0.1 * np.eye(dim)

    H = np.eye(dim)
    xx = np.array([0, 0], dtype=float)
    P = np.eye(dim)

    kf = KalmanFilter(F, B, H, Q, R, P, x)
    Trace = []
    mse_true_trajectory = 0
    #mse_data = 0

    for t in range(100):
        try:
            x = F @ x + B @ u + np.linalg.cholesky(Q) @ np.random.normal(0, 3, dim)
            z = H @ x + np.linalg.cholesky(R) @ np.random.normal(0, 4, dim)

            x_true = kf.predict(u)
            kf.update(z)
            #predict 
            #xp = (F @ xx) # + (B @ u)
            #P = ((F @ P) @ F.T) + Q
            xp, P =  execute(predict,[ xx, F, P, Q])
            # update
            y = z - (H @ xp)  # Measurement residual
            S = (H @ (P @ H.T)) + R  # Residual covariance
            K = ((P @ H.T) @ np.linalg.inv(S))  # Kalman gain
            xx = xp + (K @ y)
            I = np.eye(F.shape[0])
            P = ((I - (K @ H) )@ P)
            #mse_data = x - xp # we do not want to output the input points
            arr = np.array([0, 0])
            if x.shape != xp.shape or xp.shape != arr.shape:
                return np.inf
            Trace.append((x, z, xx))
            if (x_true - xp)@((x_true - xp).T) < 0 :
                exit(1)
            mse_true_trajectory =  np.add(mse_true_trajectory, ((x_true - xp)@((x_true - xp).T) )/100).real 
        except Exception as e:
            return np.inf    
    #if mse_data.any()  < 0.00000001:
    #    return np.inf
    x, z, xx = zip(*Trace)
    #if mse_true_trajectory < 0 :
    #    exit(1)
    return mse_true_trajectory


def execute(gen, x):
    return gp.execute(g, gen, x)


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

def matmul(inp, args):
    return inp[0] @ inp[1] 
def add(inp, args):
    return inp[0] + inp[1] 
def transpose(inp, args):
    return inp[0].T

if __name__ == "__main__":

    multiprocessing.freeze_support()

    #x0 = 56, 40, 8, 24, 48, 48, 40, 16
    g.nodes = matmul, add, transpose
    g.names = "matmul", "add", "transpose"
    g.arity = 2, 2, 1
    g.args = 0, 0, 0
    # input, maximum node, output, arity, parameters
    g.i = 4 # z
    g.n = 5
    g.o = 2 # x
    g.a = 4
    g.p = 0


    #xp, P =  execute(predict,[ xx, F, P, Q])
    predict0 = gp.build(
        g,
        #  0     1    2    3   4          5             6       7        8    9   10
        ["i0", "i1","i2","i3","matmul","matmul", "transpose","matmul", "add","o0","o1"],#
        [
            (1, 4), # (F @ xx)
            (0, 4),
            (1, 5), # (F @ P)
            (2, 5),
            (1, 6), # F.T
            (5, 7), # ((F @ P) @ F.T)
            (6, 7),
            (7, 8), # ((F @ P) @ F.T) + Q
            (3, 8),
            (4, 9),
            (8, 10)
        ],  # o1
        [])


    
    
    print("predict0 : ",fun(predict0))

    xx = [example() for i in range(10)]
    
    cost0 =  0 #fun(predict0)
    cost00 = 0 #fun(predict0)
    best =  np.inf, None, None
    generation = 0
    max_generation = 300000
    generated_examples_counter = 0
    while True:
        
        timeout = 100000
        timeout_counter = 0
        genes_forward = []
        for i in range(100):
            while True:
                forward = gp.rand(g)
                #backward = gp.rand(g)
                if not seen(forward):
                    genes_forward.append(forward)
                    break
                if timeout_counter > timeout and len(genes_forward) > 1:
                    break
                timeout_counter +=1
                

        
            
        costs = []
        with multiprocessing.Pool() as pool:
            costs = pool.map(fun, zip(genes_forward))

        for i, cost  in enumerate(costs):
            forward = genes_forward[i]
            cost = fun(forward)
           
            if cost < best[0]:
                pathlib.Path("split2.forward.gv").write_text(
                    gp.as_graphviz(g, forward))
                sys.stdout.write("forward\n" + gp.as_string(g, forward))
                #sys.stdout.write("backward\n" +
                #                gp.as_string(g, backward, All=True))
                pathlib.Path("Images5/split2.forward"+str(generated_examples_counter)+".gv").write_text(
                gp.as_graphviz(g, forward))
                sys.stdout.write("\n")
                best = cost, forward, "backward"
                print("best : ",best[0])
                generated_examples_counter += 1
            generation += 1
            if generation % 10000 == 1 or generation == max_generation:
                
                cost_w=fun(best[1])
                sys.stdout.write(
                    f"{cost_w:.16e}{cost00:.16e}\n")
            if generation == max_generation:
                gp.as_image(g,best[1],"./best.png")
                break

    #Some postprocessing stuff
    #def fun_compress(forward, backward,i):
    #     ###print("  fun_compress  ")
    #     cost = []
    #     for x0 in xx:
    #         y = execute(forward, x0)
    #         idx = np.argsort(np.abs(y[1::2]))
    #         for iiii in range(i):
    #             y[2 * idx[iiii] + 1] = 0
    #         x = execute(backward, y)
    #         l = diff(x, x0)
    #         cost.append(l)
    #     ans = statistics.mean(cost)
    #     return ans

    #compress_Haar=[]
    #compress_learned=[]

    #for i in range(50):
    #     compress_Haar.append(fun_compress(forward0,backward0,i))
    #     compress_learned.append(fun_compress(best[1],best[2],i))