import scipy.linalg
import gp
import sys
import numpy as np
import random
import subprocess
import statistics
from numpy.linalg import inv
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


class g:
    pass


def seen(a):
    a = a.tobytes()
    ans = a in Hash
    if not ans:
        Hash.add(a)
    return ans


def rand():
    while 1:
        gen = gp.rand(g)
        if not seen(gen):
            return gen


def mutate(i, top_candidate, Hash):
    mutate_prob = 0.2
    n_mutations = random.randint(1, max_mutations)
    new_genes = [top_candidate]
    for i in range(1, g.lmb + 1):
        new_candidate = top_candidate.copy()
        for m in range(n_mutations):
            j = random.randrange(g.n)
            k = random.randrange(1 + g.a + g.p)
            if k == 0:
                new_candidate[g.i + j, 0] = random.randrange(len(g.nodes))
            elif k <= g.a:
                new_candidate[g.i + j, k] = random.randrange(g.i + j)
            else:
                new_candidate[g.i + j, k] = random.randrange(g.max_val)
        for k in range(g.o):
            if random.random() < mutate_prob:
                new_candidate[g.i + g.n + k, 1] = random.randrange(g.i + g.n)
        if not seen(new_candidate):
            new_genes.append(new_candidate)
    return new_genes


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
        self.x = (self.F @ self.x)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y = z - self.x
        self.S = self.P + self.R
        self.K = self.P @ np.linalg.inv(self.S)
        self.x = self.x + (self.K @ self.y)
        self.P = (self.P - self.K @ self.P)
        return self.x


def fun(predict):
    xx = np.array([0, 0], dtype=float)
    mse_true_trajectory = 0.0
    P = np.eye(dim)
    diff = []
    try:
        rn = gp.reachable_nodes(g, predict)
    except Exception:
        return float('inf')

    kf = KalmanFilter(F, B, H, Q, R, P, x=np.array([0, 0], dtype=float))

    for x, z in traj:
        try:
            xp, P = execute(predict, [xx, F, P, Q])
            if xp.shape != (dim, ) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isnan(P)) or \
               np.any(np.isinf(xp)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')
            y = z - (H @ xp)
            S = H @ (P @ H.T) + R
            K = (P @ H.T) @ np.linalg.inv(S)
            xx = xp + (K @ y)
            I = np.eye(dim)
            P = (I - (K @ H)) @ P

            x_true = kf.predict(u)
            kf.update(z)

            if x_true.shape != xp.shape:
                return float('inf')

            diff_current = x - xx
            if np.any(np.isinf(diff_current)) or np.any(
                    np.isnan(diff_current)):
                return float('inf')

            diff.append(diff_current @ diff_current.T)

        except (ValueError, TypeError, np.linalg.LinAlgError, OverflowError,
                FloatingPointError):
            return float('inf')

    loss = np.mean(diff)
    return loss if not math.isnan(loss) else float('inf')


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

g.nodes = (matmul, add, transpose)
g.names = ("matmul", "add", "transpose")
g.arity = (2, 2, 1)
g.args = (0, 0, 0)
g.i = 4  # Input count
g.n = 5  # Number of internal nodes
g.o = 2  # Output count
g.a = 4
g.p = 0
Hash = set()

# Sanity check
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

print("Sanity check loss:", fun(predict0))

if __name__ == "__main__":
    multiprocessing.freeze_support()

    Hash = set()
    dtype = float
    random.seed(time.time())

    batch_size = 1000
    best_score = float('inf')
    best_gene = None
    evaluations = 0

    with multiprocessing.Pool() as pool:
        while True:
            # Generate unique random genes
            genes = []
            while len(genes) < batch_size:
                gene = gp.rand(g)
                if not seen(gene):
                    genes.append(gene)
            # Evaluate in parallel
            scores = pool.map(fun, genes)

            for gene, score in zip(genes, scores):
                evaluations += 1
                if score < best_score:
                    best_score = score
                    best_gene = gene
                    sys.stdout.write(
                        f"[{evaluations:05}] New best score: {best_score:.4e}\n"
                    )
                    sys.stdout.write(
                        f"Graph: {gp.as_graphviz(g, best_gene)}\n")
                    sys.stdout.flush()
