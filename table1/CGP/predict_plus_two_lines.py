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
import matplotlib.pyplot as plt
import os

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt


def kalman_update(xp, P, y, S, H):
    """Applies one Kalman-style update step."""
    K = (P @ H.T) @ np.linalg.inv(S)
    x_est = xp + (K @ y)
    P_updated = (np.eye(P.shape[0]) - (K @ H)) @ P
    return x_est, P_updated


def visualize_prediction(predict, traj, filename="position_velocity_plot.png", title="Position vs Velocity (Phase Space)", use_half=False):
    x_est = np.array([0.0, 0.0], dtype=float)
    P = np.eye(dim, dtype=float)
    xx_list = []
    x_true_list = []
    kf_list = []
    obs_list = []
    xp_list = []

    error_to_oracle = []
    error_to_kalman = []
    error_pred_vs_true = []

    kf = KalmanFilter(F.copy(), B.copy(), H.copy(), Q.copy(), R.copy(), P.copy(), x=x_est.copy())
    loop_traj = traj[:len(traj)//2] if use_half else traj

    for x_true, z in loop_traj:
        try:
            xp, P_new, y, S = execute(predict, [
                x_est.copy(), F.copy(), P.copy(), Q.copy(), z.copy(), R.copy()
            ])

            if xp.shape != (dim,) or P_new.shape != (dim, dim):
                break
            if np.any(np.isnan(xp)) or np.any(np.isinf(xp)) or \
               np.any(np.isnan(P_new)) or np.any(np.isinf(P_new)):
                break
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P_new) > 1e6:
                break

            # Kalman-style update (via helper)
            x_est, P = kalman_update(xp, P_new, y, S, H)

            xx_list.append(x_est.copy())
            xp_list.append(xp.copy())
            x_true_list.append(x_true.copy())
            obs_list.append(z.copy())

            # Kalman oracle trajectory
            xp_kalman = kf.predict(u)
            kf.update(z)
            x_kf = kf.x.copy()
            kf_list.append(x_kf)

            # Compute squared errors
            err_oracle = x_true - x_est
            err_kalman = xp_kalman - xp
            err_pred = x_true - xp

            e1 = np.dot(err_oracle, err_oracle)
            e2 = np.dot(err_kalman, err_kalman)
            e3 = np.dot(err_pred, err_pred)

            if np.isfinite(e1):
                error_to_oracle.append(e1)
            if np.isfinite(e2):
                error_to_kalman.append(e2)
            if np.isfinite(e3):
                error_pred_vs_true.append(e3)

        except Exception as e:
            print(f"Error during prediction: {e}")
            break

    if not xx_list:
        print("Warning: no valid data for plotting.")
        return

    # Convert to arrays
    xx_array = np.array(xx_list)
    xp_array = np.array(xp_list)
    x_true_array = np.array(x_true_list)
    kf_array = np.array(kf_list)
    obs_array = np.array(obs_list)

    mse_oracle = np.mean(error_to_oracle) if error_to_oracle else float('inf')
    mse_kalman = np.mean(error_to_kalman) if error_to_kalman else float('inf')
    mse_pred_vs_true = np.mean(error_pred_vs_true) if error_pred_vs_true else float('inf')

    # === Phase Space Plot ===
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x_true_array[:, 0], x_true_array[:, 1], '--', label="Oracle (True Trajectory)")
    plt.plot(xx_array[:, 0], xx_array[:, 1], '-', label="Predicted (Evolved Function)")
    plt.plot(kf_array[:, 0], kf_array[:, 1], ':', label="Kalman Filter")
    plt.scatter(obs_array[:, 0], obs_array[:, 1], s=20, c='red', edgecolors='black', linewidths=0.3, alpha=0.8, label="Observations")
    loss_text = f"Loss to Oracle: {mse_oracle:.6f}, Loss to KF: {mse_kalman:.6f}, Loss xp-True: {mse_pred_vs_true:.6f}"
    plt.plot([], [], ' ', label=loss_text)
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename))
    plt.close()
    print(f"Trajectory plot saved to: plots/{filename}")

    # === Error Over Time Plot ===
    os.makedirs("losses", exist_ok=True)
    plt.figure(figsize=(10, 4))

    if error_to_oracle:
        plt.plot(error_to_oracle, label="Squared Error to Oracle", color='blue')
        plt.axhline(mse_oracle, color='blue', linestyle='--', alpha=0.4, label=f"Avg Oracle MSE: {mse_oracle:.4f}")

    if error_to_kalman:
        plt.plot(error_to_kalman, label="Squared Error to Kalman Filter", color='green')
        plt.axhline(mse_kalman, color='green', linestyle='--', alpha=0.4, label=f"Avg KF MSE: {mse_kalman:.4f}")

    if error_pred_vs_true:
        plt.plot(error_pred_vs_true, label="Squared Error: xp vs True", color='orange')
        plt.axhline(mse_pred_vs_true, color='orange', linestyle='--', alpha=0.4, label=f"Avg xp vs True MSE: {mse_pred_vs_true:.4f}")

    plt.xlabel("Time Step")
    plt.ylabel("Squared Error")
    plt.title("Filtered Error Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    loss_filename = f"loss_over_time_kalman_loss_{mse_oracle:.4f}.png"
    loss_plot_path = os.path.join("losses", loss_filename)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Filtered loss plot saved to: {loss_plot_path}")

    # === Print all average MSEs ===
    print(f"Average MSE to Oracle       : {mse_oracle:.6f}")
    print(f"Average MSE to Kalman Filter: {mse_kalman:.6f}")
    print(f"Average MSE (xp vs x_true)  : {mse_pred_vs_true:.6f}")




def minus(inp, args):
    return inp[0] - inp[1]

def matmul(inp, args):
    return inp[0] @ inp[1]

def add(inp, args):
    return inp[0] + inp[1]

def transpose(inp, args):
    return inp[0].T


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
    n_mutations = max_mutations # random.randint(1, max_mutations) # 
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
        self.P = ((I - (self.K @ self.H)) @ self.P)
        return self.x


def distance_from_kalman_filter(predict):
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
            xp, P, y, S = execute(predict, [xx, F, P, Q, z, R])
            if xp.shape != (dim,) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isnan(P)) or \
               np.any(np.isinf(xp)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')

            #y = z - xp
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
            diff.append(diff_current @ diff_current.T)

        except (ValueError, TypeError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
            return float('inf')

    loss = np.mean(diff)
    return loss if not math.isnan(loss) else float('inf')

def distance_from_target_function(predict, alpha=1.0):
    """
    Computes a combined loss:
    loss = alpha * MSE(post-update) + (1 - alpha) * MSE(pre-update)

    Args:
        predict: the evolved function
        alpha (float): weight for post-update loss

    Returns:
        float: combined loss or inf on failure
    """
    x_est = np.array([0.0, 0.0])
    P = np.eye(dim)
    squared_errors = []
    pred_errors = []

    try:
        _ = gp.reachable_nodes(g, predict)
    except Exception:
        return float('inf')

    kf = KalmanFilter(F, B, H, Q, R, P.copy(), x=x_est.copy())

    for x_true, z in traj:
        try:
            xp, P, y, S = execute(predict, [x_est, F, P, Q, z, R])

            if xp.shape != (dim,) or P.shape != (dim, dim):
                return float('inf')
            if np.any(np.isnan(xp)) or np.any(np.isinf(xp)) or \
               np.any(np.isnan(P)) or np.any(np.isinf(P)):
                return float('inf')
            if np.linalg.norm(xp) > 1e6 or np.linalg.norm(P) > 1e6:
                return float('inf')

            # Raw prediction error
            diff_current = x - xx
            if np.any(np.isinf(diff_current)) or np.any(np.isnan(diff_current)):
                return float('inf')
            diff.append(diff_current @ diff_current.T)


            # Kalman-style update
            K = (P @ H.T) @ np.linalg.inv(S)
            x_est = xp + (K @ y)
            P = (np.eye(dim) - (K @ H)) @ P

            kf.predict(u)
            kf.update(z)

            # Updated prediction error
            error = x_true - x_est
            if error.shape != (dim,) or np.any(np.isnan(error)) or np.any(np.isinf(error)):
                return float('inf')

            squared_errors.append(np.dot(error, error))

        except Exception:
            return float('inf')

    if not squared_errors or not pred_errors:
        return float('inf')

    # Combine losses
    mse_updated = np.mean(squared_errors)
    mse_pred = np.mean(pred_errors)
    combined_loss = alpha * mse_updated + (1 - alpha) * mse_pred
    return combined_loss



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


g.nodes = (matmul, minus, add, transpose)
g.names = ("matmul","minus","add","transpose")
g.arity = (2,2,2,1)
g.args = (0,0,0,0)

g.i = 6
g.n = 7
g.o = 4
g.a = 2
g.p = 0
g.lmb = 1000





predict0 = gp.build(
        g,
        #  0     1    2    3   4    5    6        7        8           9        10       11     12      13      14    15   16   17   
        ["i0", "i1","i2","i3","i4","i5","matmul","matmul", "transpose","matmul", "add","minus","add","o0","o1","o2","o3"],#
        [
            (1, 6), # x = (F @ xx)
            (0, 6),
            (1, 7), # (F @ P)
            (2, 7),
            (1, 8), # F.T
            (7, 9), # ((F @ P) @ F.T)
            (8, 9),
            (9, 10), # P = ((F @ P) @ F.T) + Q 
            (3, 10),
            (4, 11), # y = z - self.x
            (6, 11),
            (10, 12), # R + P
            (5, 12),         
            (6, 13),
            (10, 14),
            (11, 15),
            (12, 16)
        ],
        [])


print("Sanity check loss:", distance_from_target_function(predict0), "distance from kalman filter : ",distance_from_kalman_filter(predict0))



if __name__ == "__main__":
    
    multiprocessing.freeze_support()

    
    Hash = set()
    dtype = float
    random.seed(time.time())
    

    num_islands = 4
    island_population = g.lmb + 1
    island_generations = 50
    max_generation = 100000
    max_mutations = 30 * g.n * (1 + g.a + g.p) // 100

    pool = multiprocessing.Pool()

    islands = []
    for _ in range(num_islands):
        sampler = TopCandidateSampler(max_candidates=50)
        population = [rand() for _ in range(island_population)]
        genes = population
        costs = pool.map(distance_from_target_function, genes)
        sampler.update(genes, costs)
        islands.append({"sampler": sampler, "population": population})

    generation = 0

    while generation < max_generation:
        for island_idx, island in enumerate(islands):

            sampler = island["sampler"]
            top_candidate = sampler.sample(temperature=2)
            i_best = np.argmin(costs)

            if generation % 50 == 0:
                sys.stdout.write(f"Generation {generation:05} - Island {island_idx} Best found in this generation: {costs[i_best]}\n")
                best_gene, best_score = sampler.best()
                visualize_prediction(best_gene, traj, filename="predict"+str(best_score)+"_vs_kalman.png")
                sys.stdout.write(f"Island {island_idx} BEST SO FAR: {best_score} DISTANCE FROM KALMAN FILTER : {distance_from_kalman_filter(best_gene)} \n")
                sys.stdout.write(f"Island {island_idx} BEST Graph : { gp.as_graphviz(g, best_gene)}\n")
                sys.stdout.flush()

            if costs[i_best] == float('inf'):
                island["population"] = [rand() for _ in range(island_population)]
            else:
                island["population"] = mutate(i_best, top_candidate, Hash)
            genes = island["population"]
            costs = pool.map(distance_from_target_function, genes)
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
                costs = pool.map(distance_from_target_function, genes)
                islands[worst_idx]["sampler"] = TopCandidateSampler(max_candidates=50)
                islands[worst_idx]["sampler"].update(genes, costs)
                