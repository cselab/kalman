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

def inv(inp, args):
    return np.linalg.inv(inp[0])


class TopCandidateSampler:
    def __init__(self, max_candidates=0.2):
        self.max_candidates = max_candidates
        self._heap = []
        self.counter = 0

    def update(self, new_candidates, new_scores):
        for gene, score in zip(new_candidates, new_scores):
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
        probs /= np.sum(probs)
        idx = np.random.choice(len(self._heap), p=probs)
        return self._heap[idx][2]

    def best(self):
        if not self._heap:
            raise ValueError("No candidates stored yet.")
        best_entry = max(self._heap, key=lambda x: x[0])
        return best_entry[2], -best_entry[0]

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
    n_mutations = random.randint(1, max_mutations) #  max_mutations # 
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

    def predict(self, u=np.zeros((1,1))):
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y = z - (self.H @ self.x)
        self.S = (self.H @ (self.P @ self.H.T)) + self.R
        self.K = ((self.P @ self.H.T) @ np.linalg.inv(self.S))
        self.x = self.x + (self.K @ self.y)
        I = np.eye(self.F.shape[0])
        self.P = self.P- (self.K @ self.P)
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
            xp, P, y, S,  K, xx  = execute(predict, [xx, F, P, Q, z, R])
            if xp.shape != (dim,) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isnan(P)) or \
               np.any(np.isinf(xp)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')

            #y = z - xp
            #S = H @ (P @ H.T) + R
            #K = (P) @ np.linalg.inv(S)
            #xx = xp + (K @ y)
            #I = np.eye(dim)
            #P = (I - (K @ H)) @ P

            x_true = kf.predict(u)
            kf.update(z)

            if x_true.shape != xp.shape:
                return float('inf')

            diff_current = x_true - xp
            if np.any(np.isinf(diff_current)) or np.any(np.isnan(diff_current)):
                return float('inf')
            diff.append(diff_current @ diff_current.T)

        except (ValueError, TypeError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
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


g.nodes = (matmul, minus, add, transpose, inv)
g.names = ("matmul","minus","add","transpose","inv")
g.arity = (2,2,2,1,1)
g.args = (0,0,0,0,0)

g.i = 6
g.n = 17
g.o = 6
g.a = 2
g.p = 0
g.lmb = 1000





predict0 = gp.build(
        g,
        #  0     1    2    3   4    5    6        7        8           9        10       11     12    13   14        15       16    17       18      19  20    21   22   23   24
        ["i0", "i1","i2","i3","i4","i5","matmul","matmul", "transpose","matmul", "add","minus","add","inv","matmul","matmul","add","matmul","minus","o0","o1","o2","o3","o4","o5"],#
        [
            (1, 6), # x = (F @ xx)
            (0, 6),
            (1, 7), # (F @ P)
            (2, 7),
            (1, 8), # F.T
            (7, 9), #  ((F @ P) @ F.T)
            (8, 9),
            (9, 10), # P = ((F @ P) @ F.T) + Q 
            (3, 10),
            (4, 11), # y = z - self.x
            (6, 11),
            (10, 12), # S = R + P
            (5, 12),
            (12,13),  # K = (P) @ np.linalg.inv(S)         
            (10,14),
            (13,14),
            (14,15),  # xx = xp + (K @ y)
            (11,15),  
            (6,16),
            (15,16),
            (14,17),  #P = P - (K @ P)
            (10,17),
            (10,18),
            (17,18),            
            (6, 19),
            (18, 20),
            (11, 21),
            (12, 22),
            (14, 23),
            (16, 24),
        ],
        [])


print("Sanity check loss :", fun(predict0))
print("Solution :",gp.as_graphviz(g, predict0))


if __name__ == "__main__":
    
    multiprocessing.freeze_support()

    
    Hash = set()
    dtype = float
    random.seed(time.time())
    

    num_islands = 4
    island_population = g.lmb + 1
    island_generations = 50
    max_generation = 100000
    max_mutations = 50 * g.n * (1 + g.a + g.p) // 100

    pool = multiprocessing.Pool()

    islands = []
    for _ in range(num_islands):
        sampler = TopCandidateSampler(max_candidates=50)
        population = [rand() for _ in range(island_population)]
        genes = population
        costs = pool.map(fun, genes)
        sampler.update(genes, costs)
        islands.append({"sampler": sampler, "population": population})

    generation = 0

    while generation < max_generation:
        for island_idx, island in enumerate(islands):

            sampler = island["sampler"]
            top_candidate = sampler.sample(temperature=2)
            i_best = np.argmin(costs)

            if generation % 50 == 0:
                sys.stdout.write(f"Generation {generation:05} - Island {island_idx} Best Cost: {costs[i_best]:.4e}\n")
                best_gene, best_score = sampler.best()
                sys.stdout.write(f"Island {island_idx} BEST SO FAR: {best_score:.4e}\n")
                sys.stdout.write(f"Island {island_idx} BEST Graph : { gp.as_graphviz(g, best_gene)}\n")
                sys.stdout.flush()

            if costs[i_best] == float('inf'):
                island["population"] = [rand() for _ in range(island_population)]
            else:
                island["population"] = mutate(i_best, top_candidate, Hash)
            genes = island["population"]
            costs = pool.map(fun, genes)
            sampler.update(genes, costs)

        generation += 1

        if generation % island_generations == 0:
            sys.stdout.write("\n--- Resetting half of the worst islands ---\n")
            scores_with_index = []
            for idx, island in enumerate(islands):
                try:
                    _, score = island["sampler"].best()
                    scores_with_index.append((score, idx))
                except ValueError:
                    scores_with_index.append((float('inf'), idx))

            scores_with_index.sort()
            half = len(islands) // 2
            best_half = scores_with_index[:half]
            worst_half = scores_with_index[half:]

            best_half_indices = [idx for _, idx in best_half]
            sys.stdout.write(f"Best half island indices: {best_half_indices}\n")

            for (_, worst_idx), (best_gene, best_idx) in zip(
                worst_half,
                [(islands[i]["sampler"].best()[0], i) for _, i in best_half]
            ):
                sys.stdout.write(f"Resetting Island {worst_idx} using best from Island {best_idx}\n")
                islands[worst_idx]["population"] = mutate(0, best_gene, Hash)
                genes = islands[worst_idx]["population"]
                costs = pool.map(fun, genes)
                islands[worst_idx]["sampler"] = TopCandidateSampler(max_candidates=50)
                islands[worst_idx]["sampler"].update(genes, costs)
                
                

