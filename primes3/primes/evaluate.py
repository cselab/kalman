import sys
import numpy as np
import subprocess
import traceback
import os
import tempfile

# === Load Ground Truth ===
nth_max = 100_000_000
true_values = np.load("prime_list.npy", mmap_mode='r')[:nth_max]

# === Evaluate Heuristic Function ===
def evaluate_graph(heuristic_code: str) -> float:
    code_file = None
    eval_file = None
    truth_file = None

    try:
        # Save the true values to a .npy file for subprocess access
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tf:
            np.save(tf, true_values)
            truth_file = tf.name

        # Save the heuristic function code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
            code_file.write(heuristic_code)
            code_file.flush()
            code_path = code_file.name

        # Build the evaluation script as a string (vectorized)
        eval_script = f"""
import numpy as np

def create_function(path):
    try:
        with open(path, 'r') as f:
            code = f.read()
        local_vars = {{}}
        global_vars = {{"np": np}}
        exec(code, global_vars, local_vars)
        return local_vars['heuristic']
    except Exception:
        return None

def evaluate():
    true_values = np.load('{truth_file}', mmap_mode='r')
    heuristic = create_function('{code_path}')
    if heuristic is None:
        print("inf", flush=True)
        return

    try:
        domain = np.arange(2, len(true_values) + 2)

        # === Vectorized Evaluation ===
        preds = heuristic(domain)

        if preds.shape != true_values.shape:
            print("inf", flush=True)
            return

        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            print("inf", flush=True)
            return

        error = true_values - preds
        mse = np.mean(error ** 2)
        print(mse, flush=True)
    except Exception as e:
        print("inf", flush=True)

if __name__ == "__main__":
    evaluate()
"""

        # Save evaluation script to a file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as eval_file:
            eval_file.write(eval_script)
            eval_file.flush()

        # Run the subprocess
        process = subprocess.run(
            [sys.executable, eval_file.name],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            timeout=600  # allow plenty of time
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
        if truth_file:
            try:
                os.unlink(truth_file)
            except Exception:
                pass


# === Example Usage ===
if __name__ == "__main__":
    my_heuristic = """
def heuristic(n):
    from numpy import log as ln
    return n * (ln(n) + ln(ln(n)) - 1 + ((ln(ln(n)) - 2) / ln(n)))
"""
    result = evaluate_graph(my_heuristic)
    print("MSE:", result)
