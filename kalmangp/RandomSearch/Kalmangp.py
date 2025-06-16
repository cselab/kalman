import multiprocessing
import sys
import time
import random
import pathlib
import numpy as np
import gp  # your genetic programming module

##############################################################################
# Your existing code pieces
##############################################################################

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

    def predict(self, u=np.zeros((1,1))):
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
    
    mse_true_trajectory = 0.0

    for _ in range(2000):
        try:
            # Simulate the true system
            x = F @ x + B @ u + np.linalg.cholesky(Q) @ np.random.normal(0, 3, dim)
            z = H @ x + np.linalg.cholesky(R) @ np.random.normal(0, 4, dim)
            
            # Kalman filter steps
            x_true = kf.predict(u)
            kf.update(z)
            
            # Manual update step via the GP-coded function
            xp, P =  execute(predict, [xx, F, P, Q])
            y = z - (H @ xp)
            S = H @ (P @ H.T) + R
            K = (P @ H.T) @ np.linalg.inv(S)
            xx = xp + (K @ y)
            
            I = np.eye(dim)
            P = (I - (K @ H)) @ P
            
            # Check dimension consistency
            if x.shape != x_true.shape or x_true.shape != xp.shape:
                return np.inf
            
            # Accumulate MSE between x_true and xp
            diff = x_true - xp
            if diff @ diff.T < 0:
                return np.inf
            mse_true_trajectory += (diff @ diff.T) / 1000.0
        except Exception as e:
            return np.inf
    return mse_true_trajectory

##############################################################################
# Prepare the gene representation
##############################################################################

# Example node definitions
def matmul(inp, args):
    return inp[0] @ inp[1] 

def add(inp, args):
    return inp[0] + inp[1] 

def transpose(inp, args):
    return inp[0].T

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


# Build a default gene as an example
predict0 = gp.build(
    g,
    ["i0", "i1","i2","i3","matmul","matmul","transpose","matmul","add","o0","o1"],
    [
        (1, 4),   # (F @ xx)
        (0, 4),
        (1, 5),   # (F @ P)
        (2, 5),
        (1, 6),   # F.T
        (5, 7),   # ((F @ P) @ F.T)
        (6, 7),
        (7, 8),   # ((F @ P) @ F.T) + Q
        (3, 8),
        (4, 9),
        (8, 10)
    ],
    []
)
print("Initial predict0 cost:", fun(predict0))

##############################################################################
# Parallel search logic
##############################################################################

def worker_func(
    shared_best_score,
    shared_best_gene,
    generation_counter,
    max_generation,
    lock,
    stop_event
):
    """
    Each worker:
      - Checks if the global generation limit is reached.
      - Generates new 'forward' genes until it finds one not in 'seen_genes'.
      - Uses double-check lock to update the global best if improved.
      - Increments generation counter each time a new gene is evaluated.
    """
    Hash = set()
    local_best_score = np.inf

    while not stop_event.is_set():
        # 1) Check generation limit (quick lock).
        with lock:
            if generation_counter.value >= max_generation:
                stop_event.set()
                break
         

        # 2) Generate a new gene not in seen_genes
        forward = None
        timeout = 0
        while forward is None:
            candidate = gp.rand(g)
            if not seen(Hash,candidate) :
                forward = candidate
        # 4) Compute cost outside the lock
        cost = fun(forward)
        
        # 5) Re-acquire lock and check if still better
        if cost < local_best_score:
            with lock:
                # Re-check; something else may have updated the best
                if cost < shared_best_score.value:
                    # It's still better => update best
                    shared_best_score.value = cost
                    # For storing the entire gene, store it as a gp.as_string or raw bytes
                    shared_best_gene.value = gp.as_string(g, forward)

                    local_best_score = shared_best_score.value
                    # (Optional) print or log
                    sys.stdout.write(
                        f"[Worker {multiprocessing.current_process().name}] New best: {cost}\n New graph : \n  {gp.as_string(g, forward)}\n"
                    )
                    sys.stdout.flush()
                else:
                    local_best_score = shared_best_score.value


def main_parallel_search():
    num_workers = 100  # or more if your machine can handle it
    max_generation = 300000000

    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    stop_event = multiprocessing.Event()

    # Shared variables
    shared_best_score = manager.Value('d', float('inf'))
    shared_best_gene = manager.Value('s', "")  # Enough size to store gene
    generation_counter = manager.Value('i', 0)

    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(
            target=worker_func,
            args=(
                shared_best_score,
                shared_best_gene,
                generation_counter,
                max_generation,
                lock,
                stop_event
            )
        )
        processes.append(p)
        p.start()

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Final best
    best_cost = shared_best_score.value
    best_gene_str = shared_best_gene.value

    # Optionally convert best_gene_str back to a gene object if needed:
    # best_gene_obj = gp.from_string(g, best_gene_str)  # if you have such a function

    pathlib.Path("final_best_gene.gv").write_text(best_gene_str)
    print("===== Parallel Search Complete =====")
    print(f"Total Evaluations: {generation_counter.value}")
    print(f"Best cost found: {best_cost}")
    print(f"Best gene (as string):\n{best_gene_str}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_parallel_search()
