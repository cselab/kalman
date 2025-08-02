import sys
import numpy as np
import subprocess
import traceback
import os
import tempfile

# === Generate Trajectory Once ===
dim = 2
x = np.array([0, 0], dtype=float)
F = np.array([[1, 1], [0, 1]], dtype=float)
cQ = np.array([[1 / 2, 0], [1, 0]], dtype=float)
Q = cQ @ cQ.T
cR = np.eye(dim)
R = cR @ cR.T
H = np.eye(dim)
B = np.eye(dim)
traj = []
nprng = np.random.default_rng(seed=12345)

for _ in range(200):
    x = np.array([0.05 * x[0]**3 - 2 * x[0], 0.1 * np.sin(x[1])])
    x = F @ x + cQ @ nprng.normal(0, 1, dim)
    z = H @ x + cR @ nprng.normal(0, 1, dim)
    traj.append((x.copy(), z.copy()))


# === Evaluate Graph Using Provided Python Code ===
def evaluate_graph(aproximate: str) -> float:
    code_file = None
    eval_file = None

    traj_str = "[" + ", ".join(
        f"(np.array({x.tolist()}), np.array({z.tolist()}))"
        for x, z in traj) + "]"
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as code_file:
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
    F = np.array([[1, 1], [0, 1]], dtype=float)
    cQ = np.array([[0.5, 0], [1, 0]], dtype=float)
    Q = cQ @ cQ.T
    R = np.eye(dim)
    xx = np.array([0, 0], dtype=float)
    P = np.eye(dim)

    aproximate_func = create_function('{code_path}')
    if aproximate_func is None:
        print("inf", flush=True)
        return

    squared_error = 0.0
    count = 0

    for x, z in traj:
        try:
            xp, P, y, S, K, xx = aproximate_func(xx, F, P, Q, z, R)
            diff = x - xx
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

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as eval_file:
            eval_file.write(eval_script)
            eval_file.flush()

        process = subprocess.run([sys.executable, eval_file.name],
                                 stderr=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 timeout=60)

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
    x = np.array([
        0.05 * x[0]**3-2*x[0],
        0.1 * np.sin(x[1])
    ])
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
