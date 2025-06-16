import sys
import numpy as np
import subprocess
import traceback
import os
import tempfile

# === Generate Delay-Aware Trajectory Once ===
dim = 2
dt = 1.0
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
x = np.array([0, 0], dtype=float)
traj = []
true_states = [x.copy()]
nprng = np.random.default_rng(seed=12345)

def get_F_Q(effective_dt):
    F = np.array([[1, effective_dt],
                  [0, 1]], dtype=float)
    G = np.array([[0.5 * effective_dt ** 2],
                  [effective_dt]])
    Q = G @ G.T
    return F, Q

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
    traj.append((x.copy(), z.copy(), F_dyn, Q_dyn))

# === Evaluate Graph Using Provided Python Code ===
def evaluate_graph(aproximate: str) -> float:
    code_file = None
    eval_file = None

    traj_str = "[" + ",\n".join(
        f"(np.array({x.tolist()}), np.array({z.tolist()}), np.array({F.tolist()}), np.array({Q.tolist()}))"
        for x, z, F, Q in traj
    ) + "]"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
            code_file.write(aproximate)
            code_file.flush()
            code_path = code_file.name

        eval_script = f"""
import numpy as np

traj = {traj_str}

def create_function(path):
    try:
        with open(path, 'r') as f:
            code = f.read()
        local_vars = {{}}
        global_vars = {{"np": np}}
        exec(code, global_vars, local_vars)
        return local_vars['aproximate']
    except Exception:
        return None

def evaluate():
    dim = 2
    R = np.eye(dim)
    x_est = np.zeros(dim)
    P = np.eye(dim)

    aproximate_func = create_function('{code_path}')
    if aproximate_func is None:
        print("inf", flush=True)
        return

    squared_error = 0.0
    count = 0

    for x_true, z, F_dyn, Q_dyn in traj:
        try:
            xp, P, y, S, K, x_est = aproximate_func(x_est, F_dyn, P, Q_dyn, z, R)
            diff = x_true - x_est
            if diff.shape != (dim,) or np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
                print("inf", flush=True)
                return
            squared_error += np.sum(diff ** 2)
            count += 1
        except Exception:
            print("inf", flush=True)
            return

    if count == 0:
        print("inf", flush=True)
        return

    mse = squared_error / count
    print(mse, flush=True)

if __name__ == "__main__":
    evaluate()
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as eval_file:
            eval_file.write(eval_script)
            eval_file.flush()

        process = subprocess.run(
            [sys.executable, eval_file.name],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            timeout=60
        )

        stdout_str = process.stdout.decode().strip()
        stderr_str = process.stderr.decode().strip()

        if stderr_str:
            print("⚠️ Error Output:", stderr_str)
        if not stdout_str:
            print("⚠️ No output received from subprocess.")
            return float('inf')

        try:
            result = float(stdout_str)
            return result
        except Exception as parse_err:
            print(f"⚠️ Parse error: {stdout_str!r} ({parse_err})")
            return float('inf')

    except subprocess.TimeoutExpired:
        print("Subprocess timed out!")
        return float('inf')
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return float('inf')
    finally:
        for f in [code_file, eval_file]:
            if f is not None:
                try:
                    os.unlink(f.name)
                except Exception:
                    pass

# === Example Usage ===
my_function = """
def aproximate(x, F, P, Q, z, R):
    xp = F @ x
    P = F @ P @ F.T + Q
    y = z - xp
    S = P + R
    K = P @ np.linalg.inv(S)
    x = xp + K @ y
    P = (np.eye(F.shape[0]) - K) @ P
    return xp, P, y, S, K, x
"""

result = evaluate_graph(my_function)
print("MSE:", result)