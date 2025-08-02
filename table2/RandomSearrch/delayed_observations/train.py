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
import os


def minus(inp, args):
    return inp[0] - inp[1]


def matmul(inp, args):
    return inp[0] @ inp[1]


def add(inp, args):
    return inp[0] + inp[1]


def transpose(inp, args):
    return inp[0].T


def inv(inp, args):
    return np.linalg.inv(inp[0])


class TopCandidateSampler:

    def __init__(self, max_candidates=0.2):
        self.max_candidates = max_candidates
        self._heap = []
        self.counter = 0

    def update(self, new_candidates, new_scores):
        cleaned = [(g, s) for g, s in zip(new_candidates, new_scores)
                   if np.isfinite(s) and not np.isnan(s)]
        for gene, score in cleaned:
            if score == float('inf'):
                continue
            entry = (-score, self.counter, gene)
            self.counter += 1
            if len(self._heap) < self.max_candidates:
                heapq.heappush(self._heap, entry)
            else:
                if -entry[0] < -self._heap[0][0]:
                    heapq.heappushpop(self._heap, entry)

    def sample(self, temperature=2):
        if not self._heap:
            raise ValueError("No candidates available to sample from.")
        scores = np.array([-score for score, _, _ in self._heap])
        temp = abs(temperature) if temperature != 0 else 1e-6
        scaled = -scores / temp
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        sum_probs = np.sum(probs)
        if not np.isfinite(sum_probs) or sum_probs == 0:
            probs = np.ones(len(self._heap)) / len(self._heap)
        else:
            probs /= sum_probs
        idx = np.random.choice(len(self._heap), p=probs)
        return self._heap[idx][2]

    def best(self):
        if not self._heap:
            raise ValueError("No candidates stored yet.")
        best_entry = max(self._heap, key=lambda x: x[0])
        best_score = -best_entry[0]
        if np.isnan(best_score):
            best_score = float('inf')
        if not np.isfinite(best_score):
            # Try to sample another candidate if best is still not finite
            for entry in sorted(self._heap, key=lambda x: x[0], reverse=True):
                score = -entry[0]
                gene = entry[2]
                if np.isfinite(score):
                    return gene, score
            return best_entry[2], float('inf')  # fallback to something
        return best_entry[2], best_score

    def worst_score(self):
        if not self._heap:
            return float('inf')
        return -min(self._heap, key=lambda x: -x[0])[0]


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
    n_mutations = max_mutations
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


def execute(gen, x):
    return gp.execute(g, gen, x)


# === Delay-Aware F and Q ===
def get_F_Q(effective_dt):
    F = np.array([[1, effective_dt], [0, 1]], dtype=float)
    G = np.array([[0.5 * effective_dt**2], [effective_dt]])
    Q = G @ G.T
    return F, Q


# === Trajectory Generation with Small Delays ===
dim = 2
dt = 1.0
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
cR = np.eye(dim)
H = np.eye(dim)
R = cR @ cR.T
B = np.eye(dim)
x = np.array([0, 0], dtype=float)
traj = []
true_states = [x.copy()]
nprng = np.random.default_rng(seed=12345)

for t in range(1, 200):
    delay = nprng.uniform(0.01, 0.3)
    effective_dt = dt + delay
    F_dyn, Q_dyn = get_F_Q(effective_dt)
    x = F_dyn @ x + cQ @ nprng.normal(0, 1, dim)
    true_states.append(x.copy())

    ε_idx = t - delay
    t0 = int(np.floor(ε_idx))
    t1 = min(t0 + 1, len(true_states) - 1)
    α = ε_idx - t0
    x_interp = (1 - α) * true_states[t0] + α * true_states[t1]
    z = H @ x_interp + cR @ nprng.normal(0, 1, dim)
    traj.append((x.copy(), z, F_dyn, Q_dyn))


# === Delay-Aware Fitness Function ===
def distance_from_target_function(predict, alpha=1.0):
    x_est = np.array([0.0, 0.0])
    P = np.eye(dim)
    squared_errors = []
    pred_errors = []

    try:
        _ = gp.reachable_nodes(g, predict)
    except Exception:
        return float('inf')

    for x_true, z, F_dyn, Q_dyn in traj:
        try:
            xp, P, y, S, K, x_est = execute(predict,
                                            [x_est, F_dyn, P, Q_dyn, z, R])

            if xp.shape != (dim, ) or np.any(np.isnan(xp)) or np.any(
                    np.isinf(xp)):
                return float('inf')

            pred_error = x_true - xp
            pred_errors.append(float(np.dot(pred_error, pred_error)))

            update_error = x_true - x_est
            squared_errors.append(float(np.dot(update_error, update_error)))

        except Exception:
            return float('inf')

    if not squared_errors or not pred_errors:
        return float('inf')

    mse_pred = np.mean(pred_errors)
    mse_updated = np.mean(squared_errors)
    return alpha * mse_updated + (1 - alpha) * mse_pred


if __name__ == "__main__":
    multiprocessing.freeze_support()

    Hash = set()
    dtype = float
    random.seed(time.time())

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

    predict0 = gp.build(g, [
        "i0", "i1", "i2", "i3", "i4", "i5", "matmul", "matmul", "transpose",
        "matmul", "add", "minus", "add", "inv", "matmul", "matmul", "add",
        "matmul", "minus", "o0", "o1", "o2", "o3", "o4", "o5"
    ], [(1, 6), (0, 6), (1, 7), (2, 7), (1, 8), (7, 9), (8, 9), (9, 10),
        (3, 10), (4, 11), (6, 11), (10, 12), (5, 12), (12, 13), (10, 14),
        (13, 14), (14, 15), (11, 15), (6, 16), (15, 16), (14, 17), (10, 17),
        (10, 18), (17, 18), (6, 19), (18, 20), (11, 21), (12, 22), (14, 23),
        (16, 24)], [])

    print("Sanity check loss:", distance_from_target_function(predict0))

    num_islands = 4
    island_population = g.lmb + 1
    island_generations = 50
    max_generation = 100000
    max_mutations = 30 * g.n * (1 + g.a + g.p) // 100

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
            scores = pool.map(distance_from_target_function, genes)

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
